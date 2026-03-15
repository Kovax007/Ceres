#if NO_LONGER_USED
#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;

using Ceres.Chess;

using Ceres.MCGS.Storage;
using Ceres.MCGS.Storage.Structs;
using Ceres.MCGS.Iteration.Params;

#endregion

namespace Ceres.MCGS.Search
{

  /// <summary>
  /// Applies updates to graph (Q, N, and other values for both edges and nodes)
  /// arising from selected visits.
  /// 
  /// The algorithm used depends on the BackupMethodEnum and may 
  /// use classic consecutive single leaf-to-root applications, or alternately
  /// a "reduction" algorithm which uses accumulators to merge values from multiple paths in a graph.
  /// </summary>
  public partial class MCGSBackup
  {
    public void Backup(ConcurrentQueue<MCGSPath> paths, MCGSSelectBackupStrategyBase strategy, MCGSPath path, int iteratorID)
    {
      if (Engine.Manager.ParamsSearch.TestFlag)
      {
        BackupReduced(strategy, path, iteratorID);
        return;
      }

      ref MCGSPathVisit firstVisitToProcessThisBackup = ref path.LastVisitDuringBackup;
      Debug.Assert(firstVisitToProcessThisBackup.NumVisitsAccepted <= firstVisitToProcessThisBackup.NumVisitsAttempted);

      bool isFirstTimeThisPath = path.PendingResumptionNextSlotToUse is null || path.PendingResumptionNextSlotToUse.Value == path.numSlotsUsed - 1;

      double v, d;
      int numVisitsAttempted = firstVisitToProcessThisBackup.NumVisitsAttempted;
      int numVisitsAccepted = firstVisitToProcessThisBackup.NumVisitsAccepted.Value;

      if (path.PendingResumptionNextSlotToUse is not null)
      {
        // This is beginning of a resumption, load backup values from the accumulator.
        ref MCGSPathVisit resumeVisitRef = ref firstVisitToProcessThisBackup;

        LoadLocalBackupValuesFromAccumulator(in resumeVisitRef.Accumulator, false);

        path.Iterator.DebugLogInfo("[YELLOW] RESUMPTION_RELOAD #{reload_node_index}, v={V} d={D} attempt={NumAttempted} accept={NumAccepted}",
                                   resumeVisitRef.ChildNode.Index.Index, v, d, numVisitsAccepted, numVisitsAccepted);
      }
      else if (numVisitsAccepted == 0)
      {
        v = d = double.NaN;
      }
      else
      {
        BackupValue initialBackupValue = InitialBackupValueForPath(path);
        v = initialBackupValue.V;
        d = initialBackupValue.D;
      }

      path.Iterator.DebugLogInfo("[RED] START_BACKUP_PATH [i{IteratorID}]  p{PathID} {PathSequence} {BackupMode} {TerminationReason} attempt: {NumAttempted}, accept: {NumAccepted}, RemainingPaths={remainingPaths}",
                                 path.IteratorID, path.PathID, path.PathSequenceString, path.BackupMode, (isFirstTimeThisPath ? path.TerminationReason.ToString() : "(RESUME)"),
                                 numVisitsAttempted, numVisitsAccepted, paths == null ? 0 : paths.Count);

      #region Helper method: LoadLocalBackupValuesFromAccumulator
      void LoadLocalBackupValuesFromAccumulator(ref readonly MCGSBackupAccumulator inFlightAccumulator, bool updateNumVisitsStats)
      {
        // Continue backup, but adopting the accumulated values for upward propagation.
        if (inFlightAccumulator.NumAccumulated > 0)
        {
          v = inFlightAccumulator.V;
          d = inFlightAccumulator.D;
        }
        else
        {
          v = d = 0;
        }
        if (updateNumVisitsStats)
        {
          numVisitsAttempted = inFlightAccumulator.NumVisitsAttempted;
          numVisitsAccepted = inFlightAccumulator.NumVisitsAccepted;
        }
      }
      #endregion

      bool parentIsSameSideAsEval = false;

      // Perform backup, starting from the leaf and working toward root.
      foreach (MCGSPathVisitMember visitPair in path.PathVisitsLastBackedUpToRoot)
      {
        ref MCGSPathVisit visitPathRef = ref visitPair.PathVisitRef;
        GEdge visitEdge = visitPathRef.ParentChildEdge;
        path.Iterator.DebugLogInfo("[YELLOW] START_VISIT {edge} RESUME:{PendingResumption}", visitEdge, path.PendingResumptionNextSlotToUse);

        // Clear the pending backup from this path so processing now proceeds normally.
        path.PendingResumptionNextSlotToUse = null;

        GNode parentNode = visitEdge.ParentNode;

        // STEP 1: possibly initiate prefetch of our parent (for speed)
        if (!parentNode.IsSearchRoot)
        {
          PrefetchChild(parentNode, visitPathRef.IndexOfChildInParent);
        }

        if (false && Engine.Manager.ParamsSearch.EnableGraphCatchUp && Engine.Manager.ParamsSearch.EnableGraph)
        {
          GNode thisParentNode = visitEdge.ParentNode;
          if (thisParentNode.NumParentsMoreThanOne)
          {
            foreach (GNode parent in thisParentNode.Parents)
            {
              MCGSNodeCatchUp.NodeSelfUpdate(Engine, parent);
            }
          }
        }

        // STEP 2: updated edge-related info: the edge NInFlight, W and N
        visitPathRef.NumVisitsAccepted = (short)numVisitsAccepted;

        path.Iterator.DebugLogInfo("[YELLOW] set MCGSPathVisit.NumVisitsAccepted={numVisitsAccepted} on visit {pathVisit}", numVisitsAccepted, visitPathRef);
        if (numVisitsAccepted > 0)
        {
          ProcessEdgeBackup(strategy, path,
                            ref v, ref d, ref numVisitsAccepted, parentIsSameSideAsEval, 
                            visitPair, visitPathRef, visitEdge);
        }

        // Decrement NInFlight
        GNodeStruct.UpdateEdgeNInFlightForIterator(visitEdge, iteratorID, (short)-numVisitsAttempted);
        path.Iterator.DebugLogInfo("[YELLOW] EDGE_IN_FLIGHT adjust -{NumAccepted} to {newNInFlight}", numVisitsAttempted, visitEdge.NInFlightForIterator(iteratorID));

        // If this backed up value looks to us better than a draw but
        // we know that the opponent has an available draw,
        // reset value to 0  for here and further up in tree.
        double vOurPerspective = v * (parentIsSameSideAsEval ? 1 : -1);
        if (MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST && vOurPerspective > 0 && visitEdge.ChildNodeHasDrawKnownToExist)
        {
          v = 0;
          //m = 0;
          d = 1;
        }


        // STEP 5: update parent node fields
        BackupContinueMode leafContinueMode = ProcessParentUpdate(strategy, path, ref visitPathRef,
                                                                  visitEdge.ParentNode,
                                                                  visitPathRef.IndexOfChildInParent,
                                                                  v, d,
                                                                  numVisitsAttempted, numVisitsAccepted,
                                                                  parentIsSameSideAsEval);

        switch (leafContinueMode)
        {
          case BackupContinueMode.AbortBackup:
            path.Iterator.DebugLogInfo("[YELLOW] ABORT_BACKUP");
            return;

          case BackupContinueMode.ContinueBackupWithCurrentValues:
            path.Iterator.DebugLogInfo("[YELLOW] CONTINUE_CURRENT_VALUES");
            break;

          case BackupContinueMode.ContinueBackupWithAccumulatorValues:
            // Continue backup with accumulator values.
            LoadLocalBackupValuesFromAccumulator(ref visitPathRef.Accumulator, true);
            parentIsSameSideAsEval = true;
            path.Iterator.DebugLogInfo("[YELLOW] CONTINUE_ACCUMULATOR_VALUES");
            break;

          default:
            throw new Exception("Unhandled BackupContinueMode");
        }

        parentIsSameSideAsEval = !parentIsSameSideAsEval;
      }
    }

    private static void ProcessEdgeBackup(MCGSSelectBackupStrategyBase strategy, MCGSPath path, ref double v, ref double d, ref int numVisitsAccepted, bool parentIsSameSideAsEval, MCGSPathVisitMember visitPair, MCGSPathVisit visitPathRef, GEdge visitEdge)
    {
      // Backup to edge (values have same orientation as child).
      // N.B. In the special case of a leaf visit which triggered draw-by-repetition (cycle),
      //      the edge should not be updated based on the child Q but rather this draw outcome.
      bool recomputeEdgeQ = MCGSParamsFixed.BACKUP_FULLY_SYNCHRONIZES_WITH_CHILDREN
                        && !(path.TerminationReason == MCGSPathTerminationReason.TerminalDrawByRepetition && visitPair.IsPathLeaf);

      // If the child was selected "off-policy" then we disconnect
      // disconnect the path starting at this level (and above)
      // so that the backed up value is not used and we only backout the in flight values.
      if (visitPathRef.DisconnectFromEdgeNStartingThisVisit)
      {
        path.Iterator.DebugLogInfo("[YELLOW] RESET numAccepted -> 0 because SelectionMode not OnPolicy at edge {edge}", visitEdge);
        numVisitsAccepted = 0;
        v = double.NaN;
        d = double.NaN;
      }

      if (visitPathRef.IsTranspositionExpansion && MCGSParamsFixed.SWITCH_TRANSPOSITION_EXPANSION_BACKUP_TO_TRANSPOSITION_Q)
      {
        // Determine how much weight to give to the new backed-up value versus transposition Q
        float wtChildQ = 0;
        if (visitEdge.N > 1)
        {
          float ratioChildNToParentN = (float)visitEdge.ChildNode.N / visitEdge.N;
          float ratioDeepen = path.Engine.Manager.ParamsSearch.BackupTranspositionRedescentMinMultipleToStop;

if (false && strategy.Engine.Manager.ParamsSearch.TestFlag2)
{
            // Next two lines 2+/-14

          float wtV;
          if (visitEdge.ChildNode.N < visitEdge.N + numVisitsAccepted)
          {
            wtV = 0;
          }
          else
          {
            int nLeftNew = (visitEdge.ChildNode.N - 1) - (visitEdge.N + numVisitsAccepted);
            int nRightNew = numVisitsAccepted;
            int nTotalNew = nLeftNew + nRightNew;
            wtV = (float)nRightNew / nTotalNew;
            }
            //= visitEdge.ChildNode.N > visitEdge.N ? numVisitsAccepted / ((float)visitEdge.ChildNode.N - visitEdge.N - numVisitsAccepted) : 1;
//            wtChildQ = 1.0f - wtV;
//Console.WriteLine($"TP BACKUP:  wtV={wtV:F3}  childN={visitEdge.ChildNode.N} parentN={visitEdge.N}  numVisitsAccepted={numVisitsAccepted}"); 
            // 0.3333 was -12 +/- 20
            // 0.6666 was -9 +/- 14
            //wtChildQ = 2*0.3333f;
          }
}

        double childQToUse = parentIsSameSideAsEval ? -visitEdge.ChildNode.Q : visitEdge.ChildNode.Q;
        v = (1.0f - wtChildQ) * v + (wtChildQ * childQToUse);
        d = (1.0f - wtChildQ) * d + (wtChildQ * visitEdge.ChildNode.D);
      }

      strategy.BackupToEdge(visitEdge, recomputeEdgeQ, numVisitsAccepted, v, d, !parentIsSameSideAsEval);
      Debug.Assert(visitEdge.Type.IsTerminal()
                 || !double.IsNaN(visitPathRef.ChildNode.Q)
                 || path.TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval
                 || path.TerminationReason == MCGSPathTerminationReason.PiggybackPendingNNEval);
      path.Iterator.DebugLogInfo("[YELLOW] strategy.BackupToEdge accept={NumAccepted} v={V} d={D} parentIsSameSideAsEval={ParentIsSameSideAsEval} now {edge}",
                                 numVisitsAccepted, v, d, parentIsSameSideAsEval, visitEdge);
    }

    enum BackupContinueMode
    {
      AbortBackup,
      ContinueBackupWithCurrentValues,
      ContinueBackupWithAccumulatorValues
    };

    private BackupContinueMode ProcessParentUpdate(MCGSSelectBackupStrategyBase strategy,
                                                   MCGSPath path, ref MCGSPathVisit pathVisitRef,
                                                   GNode node, int indexOfChildInParent,
                                                   double v, double d,
                                                   int numVisitsAttempted, int numVisitsAccepted,
                                                   bool sameSideAsEval)
    {
      // Any recursive upward propagation of values
      // must ignore this exactly path since it is already being updated here.
      NodeIndex parentIndexToIgnore = pathVisitRef.IndexOfPriorVisitParentNode;
      bool isMultiparent = pathVisitRef.ParentChildEdge.ParentNode.NumParentsMoreThanOne;

      #region Helper method: BackupToNode
      void BackupToNode(GNode node, int indexOfChildInParent, double v, double d, int numVisitsAccepted, bool vSameSideAsEval)
      {
        if (numVisitsAccepted > 0)
        {
          // Possibly update NumEdgesVisited (not possible in selection phase because possibly visits end up not accepted).
          if (indexOfChildInParent >= node.NumEdgesVisited)
          {
            node.NodeRef.NumEdgesVisited = (byte)(indexOfChildInParent + 1);
          }

          path.Iterator.DebugLogInfo("[YELLOW] strategy.BackupToNode #{node.Index.Index} accept={NumAccepted} v={V} d={D} same_side={SameSide}",
                                       node.Index.Index, numVisitsAccepted, v, d, sameSideAsEval);
          strategy.BackupToNode(node, numVisitsAccepted, v, d, vSameSideAsEval, Engine.Manager.ParamsSearch.NodeRecalculationPhase == MCGSPhase.Backup);
                                
          if (!node.IsSearchRoot 
            && isMultiparent
            && Engine.Manager.ParamsSearch.EnableSupplementalBackupRecompute)// ** TODO: make access here more efficient
          {
            if (Engine.Manager.ParamsSearch.Execution.BackupMode == BackupMethodEnum.LeafToRootSingleThread)
            {
              throw new Exception("EnableSupplementalBackupRecompute incompatible with LeafToRootSingleThread due to inefficiency of redundant updaing.");
            }

            const int NUM_LEVELS_UPWARD = 5;
            (strategy as MCGSStrategyPUCT).PropagateEdgeDeltaUpward(node, parentIndexToIgnore, NUM_LEVELS_UPWARD);
          }

          if (Engine.Manager.ParamsSelect.RPOBackupLambda != 0
           && node.N > Engine.Manager.ParamsSelect.RPOBackupMinN)
          {
            float lambda = Engine.Manager.ParamsSelect.RPOBackupLambda;
            if (false && strategy.Engine.Manager.ParamsSearch.TestFlag)
            {
              lambda -= node.UncertaintyPolicy * 0.35f;
            }
            RPOResult rpo = RPOTests.BestMoveInfo(node, float.NaN, node.NumEdgesVisited, 
                                                  lambda,//Coordinator.Manager.ParamsSelect.RPOBackupLambda, 
                                                  Engine.Manager.ParamsSelect.RPOLambdaPower, 
                                                  MCGSParamsFixed.RPO_USE_WEIGHTING);
            Debug.Assert(!double.IsNaN(rpo.NewQ));
            Debug.Assert(Math.Abs(rpo.NewQ) < 1.5f);

            node.NodeRef.Q = rpo.NewQ;
          }
        }
      }
      #endregion

      if (!path.BackupMode.UsesReductionAlgorithm())
      {
        // Not using reduction algorithm, perform a direct backup to the node.
        BackupToNode(node, indexOfChildInParent, v, d, numVisitsAccepted, sameSideAsEval);
        return BackupContinueMode.ContinueBackupWithCurrentValues;
      }
      else // UsesReductionAlgorithm
      {
        // In reduction mode, manage pending visits and accumulator.
        // Assert that the current path's attempted visits do not exceed the pending count before this path's processing.
        // Note: pathVisitRef.NumVisitsAttemptedPendingBackup is a live counter shared across paths.
        Debug.Assert(numVisitsAttempted <= pathVisitRef.NumVisitsAttemptedPendingBackup, $"numVisitsAttempted {numVisitsAttempted} > pathVisitRef.NumVisitsAttemptedPendingBackup {pathVisitRef.NumVisitsAttemptedPendingBackup} for node {node.Index}");

        // Accumulator has values from prior paths. Add this path's contribution.
        pathVisitRef.Accumulator.Add(sameSideAsEval ? v : -v, d, (short)numVisitsAttempted, (short)numVisitsAccepted, indexOfChildInParent);
        path.Iterator.DebugLogInfo("[YELLOW] ACCUMULATOR_ADD_ON_ZERO_PENDING #{NodeIndex} attempt={NumAttempted} accept={NumAccepted} v={V} d={D} same_side={SameSide} => att={NumAttemptedA} acc={NumAcceptedA} new_pend={newPending} v={vA} d={dA}",
                                     node.Index.Index, numVisitsAttempted, numVisitsAccepted, v, d, sameSideAsEval,
                                     pathVisitRef.Accumulator.NumVisitsAttempted, pathVisitRef.Accumulator.NumVisitsAccepted,
                                     pathVisitRef.NumVisitsAttemptedPendingBackup - numVisitsAttempted, pathVisitRef.Accumulator.V, pathVisitRef.Accumulator.D);

        int originalPendingVisits = pathVisitRef.NumVisitsAttemptedPendingBackup; // For logging
        int newNumPending = Interlocked.Add(ref pathVisitRef.NumVisitsAttemptedPendingBackup, -numVisitsAttempted);
        path.Iterator.DebugLogInfo("[YELLOW] DECREMENT_PENDING #{NodeIndex} -{att} from {origPend} becomes {newPend}",
                                   node.Index.Index, numVisitsAttempted, originalPendingVisits, newNumPending);

        if (pathVisitRef.Accumulator.NumVisitsAttempted == 0)
        {
          // Accumulator is empty or has no accepted visits. Backup this path's values directly.
          path.Iterator.DebugLogInfo("[YELLOW] ZERO_PENDING_NO_ACCUM #{NodeIndex} attempt={NumAttempted} accept={NumAccepted} v={V} d={D} same_side={SameSide}",
                                       node.Index.Index, numVisitsAttempted, numVisitsAccepted, v, d, sameSideAsEval);
          BackupToNode(node, indexOfChildInParent, v, d, numVisitsAccepted, sameSideAsEval);

          return BackupContinueMode.ContinueBackupWithCurrentValues;
        }

        if (newNumPending == 0)
        {
          // This path is the last one to process this node.
          // Backup to node using the final accumulated values.
          bool isLeafAndMultiTransposition = node.N == 0; // Preserving original logic for this specific case
          ref MCGSBackupAccumulator finalAccumulator = ref pathVisitRef.Accumulator;
          BackupToNode(node,
                       finalAccumulator.IndexOfMaxChildVisitedFromParent,
                       finalAccumulator.V,
                       finalAccumulator.D,
                       isLeafAndMultiTransposition ? 1 : finalAccumulator.NumVisitsAccepted,
                       true); // Accumulator values are always from node's perspective

          return BackupContinueMode.ContinueBackupWithAccumulatorValues;
        }
        else if (newNumPending > 0)
        {
          // Other paths are still pending for this node. Add this path's contribution to the accumulator and abort this path's backup.
          path.Iterator.DebugLogInfo("[YELLOW] ABORT #{NodeIndex} attempt={NumAttempted} accept={NumAccepted} v={V} d={D} same_side={SameSide} => att={NumAttemptedA} acc={NumAcceptedA} new_pend={newPending} v={vA} d={dA}",
                                       node.Index.Index, numVisitsAttempted, numVisitsAccepted, v, d, sameSideAsEval,
                                       pathVisitRef.Accumulator.NumVisitsAttempted, pathVisitRef.Accumulator.NumVisitsAccepted,
                                       newNumPending, pathVisitRef.Accumulator.V, pathVisitRef.Accumulator.D);
          return BackupContinueMode.AbortBackup;
        }
        else // newNumPending < 0
        {
          // This indicates an error in visit accounting.
          path.Iterator.DebugLogInfo("[RED] CRITICAL_ERROR: newNumPending ({newNumPend}) < 0 for node {NodeIndex}. PathID {PathID}. OriginalPending: {origPend}, ThisPathAttempted: {attempted}. Aborting backup for this path.",
                                      newNumPending, node.Index.Index, path.PathID, originalPendingVisits, numVisitsAttempted);
          Debug.Fail($"newNumPending ({newNumPending}) is negative for node {node.Index}, path {path.PathID}. OriginalPending: {originalPendingVisits}, ThisPathAttempted: {numVisitsAttempted}. This indicates a logic error in visit counting.");
          // Safest action is to abort this path's backup.
          return BackupContinueMode.AbortBackup;
        }
      }
    }

  }
}

#endif