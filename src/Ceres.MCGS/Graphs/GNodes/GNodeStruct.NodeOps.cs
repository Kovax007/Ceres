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
using System.Diagnostics;
using System.Threading;

using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Various support methods for GNodeStruct.
/// </summary>
public partial struct GNodeStruct
{
  public delegate float MCTSNodeStructMetricFunc(in GNodeStruct node);

  /// <summary>
  /// If this node is possibly reachable (appears as a descendent in the full game tree)
  /// from a specified prior node (using approximate heuristics).
  /// </summary>
  /// <param name="priorNode"></param>
  /// <returns></returns>
  public readonly bool IsPossiblyReachableFrom(in GNodeStruct priorNode)
    => NumPieces <= priorNode.NumPieces
    && NumRank2Pawns <= priorNode.NumRank2Pawns;


  /// <summary>
  /// Returns the MGPosition corresponding to this node.
  /// 
  /// NOTE: this is inefficient, requiring descent from root including move generation at each level.
  /// </summary>
  /// <param name="store"></param>
  /// <param name="nodeRef"></param>
  /// <returns></returns>
  public MGPosition CalcPosition(GraphStore store)
  {
    if (IsOldGeneration)
    {
      throw new Exception("Internal error: CalcPosition net yet supported for old generation nodes.");
    }

    ref readonly GNodeStruct visitorNodeRef = ref this;

    // Ascend up to root, keeping track of all moves along the way.
    Span<GNodeStruct> nodes = store.NodesStore.Span;
    Span<ushort> moves = stackalloc ushort[255];
    int index = 0;
    while (!visitorNodeRef.IsGraphRoot)
    {
      throw new NotImplementedException("next line needs remediation");
//        moves[index++] = visitorNodeRef.PriorMove.RawValue;
      visitorNodeRef = ref nodes[visitorNodeRef.ParentIndex.Index];
    }

    // Now reverse the ascent, tracking the position along the way.
    // TODO: would it be possible to sometimes or always avoid the move generation
    //       and instead use the EncodedMove directly?
    MGPosition pos = store.NodesStore.PositionHistory.FinalPosMG;
    if (index > 0)
    {
      for (int i = index - 1; i >= 0; i--)
      {
        EncodedMove move = new(moves[i]);
        MGMove moveMG = MGMoveConverter.ToMGMove(in pos, move);
        pos.MakeMove(moveMG);
      }
    }
    return pos;
  }


  #region Updates

  /// <summary>
  /// Helper method to atomically add a signed short to a ushort using a 16-bit CAS retry loop.
  /// 
  /// A 16-big CAS retry-loop is used because Interlocked primitives do not work on 16 bit types.
  /// </summary>
  /// <param name="target"></param>
  /// <param name="delta"></param>
  /// <returns></returns>
  private static ushort InterlockedAddUShort(ref ushort target, short delta)
  {
    Debug.Assert(delta != 0);

    while (true)
    {
      // ake the current snapshot (atomic because ushort is naturally aligned).
      ushort oldVal = Volatile.Read(ref target);

      // Calculate the proposed new value in a wider type.
      int newInt = oldVal + delta;
      Debug.Assert((uint)newInt <= ushort.MaxValue);

      ushort newVal = (ushort)newInt;

      // Try to swap.  If another thread beat us, retry.
      ushort observed = Interlocked.CompareExchange(ref target, newVal, oldVal);
      if (observed == oldVal)
      { 
        return newVal;               
      }

      Thread.SpinWait(0);   // tiny back-off under heavy contention
    }
  }


  /// <summary>
  /// Atomically adds delta to either NumInFlight0 or NumInFlight1.
  /// </summary>
  /// </remarks>
  public unsafe static void UpdateEdgeNInFlightForIterator(GEdge edge, int iteratorID, int adjust)
  {
    Debug.Assert(iteratorID is 0 or 1);
    Debug.Assert(adjust >= short.MinValue && adjust <= short.MaxValue);

    if (iteratorID == 0)
    {
      Debug.Assert(edge.edgeStructPtr->NumInFlight0 + adjust >= 0);
//    Interlocked.Add(ref edge.edgeStructPtr->NumInFlight0, adjust);
      InterlockedAddUShort(ref edge.edgeStructPtr->NumInFlight0, (short)adjust);
    }
    else
    {
      Debug.Assert(edge.edgeStructPtr->NumInFlight1 + adjust >= 0);
//    Interlocked.Add(ref edge.edgeStructPtr->NumInFlight1, adjust);
      InterlockedAddUShort(ref edge.edgeStructPtr->NumInFlight1, (short)adjust);
    }
  }



#if NOT
  /// <summary>
  /// Applies update to this node and all predecessors (increment N, add to W, and decrement NInFlight).
  /// </summary>
  /// <param name="vToApply"></param>
  /// <param name="mToApply"></param>
  /// <param name="numInFlight1"></param>
  /// <param name="numInFlight2"></param>
  public unsafe void BackupApply(MCTSIterator context,
                                 Span<NodeStruct> allNodes,
                                 Span<MoveInfoStruct> allChildren,
                                 VisitFromStore parentsTable,
                                 int selectorID,
                                 int numToApply,
                                 float vToApplyFirst, float vToApplyNonFirst, float mToApply,
                                 float dToApplyFirst, float dToApplyNonFirst,
                                 int numInFlight1, int numInFlight2,
                                 out NodeIndex indexOfChildDescendentFromRoot)
  {
    Debug.Assert(!float.IsNaN(vToApplyFirst));
    Debug.Assert(numInFlight2 == 0); // GFIX Disabled for MCGS

    Span<NodeStruct> nodesSpan = context.Tree.Store.Nodes.nodes.Span;

    // Tracking uncertainty boosting somewhat expensive, so only do if requested.
    bool updateUncertainty = context.ParamsSearch.EnableUncertaintyBoosting;

    indexOfChildDescendentFromRoot = default;
    bool first = true;
    float vToApply = vToApplyFirst;
    float dToApply = dToApplyFirst;
    ref NodeStruct node = ref this;
    float lastUncertaintyVAdj = float.NaN;

    while (true)
    {
      int parentIndex = node.ParentIndex.Index;

      if (!node.IsRoot)
      {
        //int ourIndexInParentsChildren = node.IndexInParent;
        Span<int> parents = stackalloc int[99];
        parentsTable.GetVisitsFrom(node.Index, parents);
        if (parents[1] == -1)
        {
          parentIndex = parents[0];
          Debug.Assert(parentIndex == node.ParentIndex.Index);
        }
        else
        {
          //            if (node.Index.Index == 68)
          //              Console.WriteLine("yesxxx");
          bool found = false;
          for (int i = 0; i < parents.Length; i++)
          {
            int piEntry = parents[i];
            if (piEntry == -1)
            {
              break;
            }

            ref NodeStruct thisParentRef = ref allNodes[piEntry];
            Span<VisitToStruct> childVisits = thisParentRef.ChildInfo.VisitsToChildren(context.Tree.Store);
            foreach (VisitToStruct vc in childVisits)
            {
              if (selectorID == 0 && vc.NumInFlight1 > 0)
              {
                found = true;
                parentIndex = piEntry;
                break;
              }
              else if (selectorID == 1 && vc.NumInFlight2 > 0)
              {
                found = true;
                parentIndex = piEntry;
                break;
              }
            }

            if (found)
            {
              break;
            }
          }
          if (!found)
          {
            throw new NotImplementedException("No parent found which was in flight, child node# " + node.Index);
          }
        }
      }
      //        NodeIndex indexOfParent = new NodeIndex(parents[0]);
      //        if (indexOfParent != node.ParentIndex)
      //        {
      //          throw new Exception("parent not matching"); // GFIX-PARENT_MATCH
      //        }


      ref NodeStruct parentRef = ref allNodes[parentIndex];

      // ** TODO:Make this conditional on the UncertaintyV adjustment feature being enabled.
      if (MCTSParamsFixed.UNCERTAINTY_TESTS_ENABLED && !first)
      {
        const float THRESHOLD = 0.08f;
        if (MathF.Pow(lastUncertaintyVAdj, 1) > THRESHOLD)
        {
          if (vToApply < (float)node.Q - 0.10f) // if much worse
          {
            float FRACTION_PARENT = 0.666f;
            float vPrior = vToApply;
            vToApply = FRACTION_PARENT * (float)node.Q
                     + (1.0f - FRACTION_PARENT) * vToApply;
            //            Console.WriteLine(vPrior + " --> " + vToApply + " because of adjustment " + lastUncertaintyVAdj + " at depth " + node.DepthInTree + " curQ:" + node.Q);
          }
        }
      }

      throw new NotImplementedException("Next line remediation");
      //lastUncertaintyVAdj = node.InfoRef.Annotation.LastUncertaintyVAdj;

      node.UpdateNInFlight(context.Tree.Store, -numInFlight1, -numInFlight2);

      // If a draw could have been claimed here, 
      // assume it would have been if the alternative was worse
      // (use value of 0 here and further up in tree)
      if (vToApply < 0 && node.DrawKnownToExistAmongChildren)
      {
        vToApply = vToApplyNonFirst = 0;
        mToApply = 0;
        dToApply = dToApplyFirst = 1; // TODO: is this ok even if not WDL network?
      }

      // Compute statistics used for tracking uncertainty (variance)
      float qDiff = vToApply - (float)node.Q;

#if LEGACY_UNCERTAINTY
      // NOTE: It is not possible to make the updates to both N and W atomic as a group.
      //       Therefore there is a very small possibility that another thread will observe one updated but not the other
      //       (e.g. thread gathering nodes which reaches over to use this as a transposition root and references Q)
      //       To mitigate the possible distortion, N is updated before W so any distortions will shrink toward 0.
      if (updateUncertainty && node.N >= MCTSNodeUncertaintyAccumulator.MIN_N_UPDATE)
      {
        node.Uncertainty.UpdateUncertainty(ref node, qDiff, numToApply, node.N);
      }
#endif

      node.N += numToApply;


      // Try to apply deep transposition backup (if requested).
      if (numToApply == 1)
      {
        node.W += vToApply;
        node.mSum += (FP16)mToApply;
        node.DSum += dToApply;
      }
      else
      {
        node.W += vToApply * numToApply;
        node.mSum += (FP16)(mToApply * numToApply);
        node.DSum += dToApply * numToApply;
      }

      // Update edge visit count.
      // GFIX: needs to be atomic? // GFIX: about 10% slowdown?
      if (!node.IsRoot)
      {
        Span<VisitToStruct> visits = parentRef.ChildInfo.VisitsToChildren(context.Tree.Store);

        if (visits[node.IndexInParent].N == 0)
        {
          // First time initialization, need to copy up the uncertainty.
          visits[node.IndexInParent].UncertaintyValue = node.ParentRef.UncertaintyValue;
          //visits[node.IndexInParent].UncertaintyPolicy = node.ParentRef.UncertaintyPolicy;
        }

        visits[node.IndexInParent].W += vToApply * numToApply;
        visits[node.IndexInParent].N += numToApply;
      }


#if FEATURE_UNCERTAINTY
      // Update uncertainty (exponentially weighted moving average)
      float absDiff = MathF.Abs(vToApply - (float)node.Q);
      node.Uncertainty = (FP16)(LAMBDA * node.Uncertainty + (1.0f - LAMBDA) * absDiff);
#endif

      if (node.IsRoot)
      {
        return;
      }
      else
      {
        if (parentRef.IsRoot)
        {
          indexOfChildDescendentFromRoot = node.Index;
          throw new NotImplementedException();//node.Context.Info.Context.RootMoveTracker.UpdateQValue(context.Tree.Root.N, node.N, node.IndexInParent, vToApply, numToApply);
        }
        else
        {
          // Initiate a prefetch of parent's parent.
          // This happens far early enough that the memory access 
          // should be complete by the time we need the data.
          PrefetchNode(ref allNodes[parentRef.ParentIndex.Index]);
        }

#if NOT
//          bool thisNodeExploratory = valuePriorAvgIsBetter < -0.03f; // -0.03f
//          bool thisNodeMuchWorse = valueThisSampleBetter < -0.20f; // -0.20f

// somewhat promising in suites but loses moderately badly in games
        if (node.N > 1 && node.ParentRef.N > 0 && !double.IsNaN(node.ParentRef.Q) && thisNodeMuchWorse && thisNodeExploratory && SearchManager.ThreadSearchContext.ParamsSearch.TEST_FLAG)
          vToApply = (float)node.Q;
#endif

        if (first)
        {
          vToApply = vToApplyNonFirst;
          dToApply = dToApplyNonFirst;
          first = false;
        }

        // Backup in tree 
        vToApply *= -1; // flip sign to change perspective
        mToApply++;     // moves left will look one greater from parent's perspective

        // Ascend to parent
        node = ref parentRef;
      }
    }
  }
#endif

#endregion


  /// <summary>
  /// Returns if node corresponds to a position with white to move.
  /// </summary>
  public bool IsWhite
  {
    readonly get => miscFields.IsWhite;
    internal set => miscFields.IsWhite = value;
    
  }


  /// <summary>
  /// Returns the new value for a variance accumulator with exponential weighting
  /// that reflects an update with a specified new update (repeated thisN times).
  /// </summary>
  static double NewEMWVarianceAcc(double priorAcc, double priorN, double squaredDeviation, int thisN, double lambda)
  {
    double newAcc = priorAcc;
    for (int i = 0; i < thisN; i++)
    {
      double priorVariance = newAcc / (priorN + i);

      double newVariance = priorVariance * (1.0f - lambda)
                        + squaredDeviation * lambda;
      newAcc = newVariance * (priorN + i + 1);
    }

    // Return the variance accumulator value which would now return 
    // our new variance target after the current sample is recorded
    return newAcc;
  }
}

#if EXPERIMENTAL
  // This is an alternate version which uses Avx2.GatherVector256
  // It seems to be correct, but in extensive tests August 2020 did not seem any faster

  public unsafe void PrefetchChildren(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, int minIndex, int maxIndex)
  {
    Debug.Assert(maxIndex - minIndex < 8);

    // The AVX instructions load from array of floats, but we are actually a
    int FLOATS_PER_NODE = sizeof(MCTSNodeStruct) / sizeof(float);

    // Prepare array of indices at which these nodes exist
    Span<int> indices = stackalloc int[8];

    Span<MCTSNodeStructChild> childSpan = store.Children.SpanForNode(in store.Nodes.Span[nodeIndex.Index]); // TODO: maybe just pass in the struct here instead of node Index?
    for (int i = minIndex; i < maxIndex; i++)
    {
      MCTSNodeStructChild child = childSpan[i];
      if (child.IsExpanded)
      {
        indices[i - minIndex] = child.ChildIndex.Index * FLOATS_PER_NODE;
      }
    }

    // Prefetch node data
    void* nodePtr = Unsafe.AsPointer(ref store.Nodes.Span[nodeIndex.Index]);
    PrefetchDataAt(nodePtr);


    fixed (int* indicesPtr = &indices[0])
    {
      Vector256<int> indicesVector = Avx2.LoadVector256(indicesPtr);
      Vector256<float> result = Avx2.GatherVector256((float*)store.Nodes.RawMemory, indicesVector, 4);
    }

  }
  // --------------------------------------------------------------------------------------------
  public void PossiblyPrefetchNodeAndChildrenInRange(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex,
                                                     int firstChildIndex, int numChildren)
  {
    int numDone = 0;
    while ((numChildren - numDone) > 3)
    {
      int numThisLoop = Math.Min(8, numChildren - numDone);
      PrefetchChildren(store, nodeIndex, firstChildIndex + numDone, firstChildIndex + numDone + numThisLoop - 1);
      numDone += numThisLoop;
    }
  }

#endif

#if EXPERIMENTAL
  public (int indexBest, int indexSecondBest) IndicesMaxExpandedChildren(MCTSNodeStructMetricFunc rankFunc)
  {
    float bestV = float.MinValue;
    int bestI = -1;
    float nextBestV = float.MinValue;
    int nextBestI = -1;
    for (int i=0; i<NumChildrenExpanded; i++)
    {
      float value = rankFunc(in ChildAtIndexRef(i));
      if (value > bestV)
      {
        nextBestV = bestV;
        nextBestI = bestI;
        bestV = value;
        bestI = i;
      }
      else if (value > nextBestV)
      {
        nextBestV = value;
        nextBestI = i;
      }
    }

    return (bestI, nextBestI);
  }
#endif


#if NOT

  /// <summary>
  /// Returns a span which re-interprets the child array as an array of FP16
  /// to be used as a V buffer.
  /// </summary>
  /// <param name="store"></param>
  /// <returns></returns>
  internal unsafe Span<FP16> ChildrenArrayAsVBuffer(MCTSNodeStore store)
  {
    // Determine how many V scores we have room for
    int maxVScores = NumPolicyMoves * (sizeof(MCTSNodeStructChild) / sizeof(FP16));
    return new Span<FP16>(Unsafe.AsPointer(ref store.Children.childIndices[ChildStartIndex]), maxVScores);
  }


  public unsafe void FillChidrenWithVScores(MCTSNodeStore store, ref MCTSNodeStruct source)
  {
    if (NumPolicyMoves > 0)
    {
      // We expect child array to e already allocated
      Debug.Assert(ChildStartBlockIndex > 0);

      // Interpet the 
      Span<FP16> vScores = ChildrenArrayAsVBuffer(store);

      int count = 0;
      MCTSNodeSequentialVisitor visitor = new MCTSNodeSequentialVisitor(store, source.Index);
      foreach (MCTSNodeStructIndex childNodeIndex in visitor.Iterate)
      {
        if (count >= vScores.Length)
          break;
        else
          vScores[count++] = store.Nodes.nodes[childNodeIndex.Index].V;
      }

      // If we did not fill V array completely, set the 
      // next element as an "end" market (NaN)
      if (count < vScores.Length) vScores[count] = FP16.NaN;
    }
  }

#endif
