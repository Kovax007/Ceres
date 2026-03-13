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
using System.Linq;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Strategies;


#endregion

namespace Ceres.MCGS.Search.PUCT;

public static class PUCTSelector
{
  /// <summary>
  /// Internal class that holds the spans in which the child statistics are gathered.
  /// </summary>
  [ThreadStatic] static GatheredChildStats gatherStats;


  /// <summary>
  /// Returns the thread static variables, initializing if first time accessed by this thread.
  /// </summary>
  /// <returns></returns>
  internal static GatheredChildStats CheckInitThreadStatics()
  {
    GatheredChildStats stats = gatherStats;
    return stats ?? (gatherStats = new GatheredChildStats());
  }


  /// <summary>
  /// Applies CPUCT selection to determine for each child
  /// their U scores and the number of visits each should receive
  /// if a specified number of total visits will be made to this node.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="node"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="selectorID"></param>
  /// <param name="rootMovePruningStatus"></param>
  /// <param name="dualCollisionFraction"></param>
  /// <param name="minChildIndex"></param>
  /// <param name="maxChildIndex"></param>
  /// <param name="numTargetVisits"></param>
  /// <param name="scores"></param>
  /// <param name="childVisitCounts"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="temperatureMultiplier"></param>
  public static NodeSelectAccumulator ComputeTopChildScores(Graph graph, GNode node,
                                                            ParamsSearch paramsSearch, ParamsSelect paramsSelect, 
                                                            int selectorID, bool refreshStaleEdges,
                                                            MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                            float dualCollisionFraction,
                                                            int minChildIndex, int maxChildIndex, int numTargetVisits,
                                                            Span<double> scores, Span<short> childVisitCounts,
                                                            float cpuctMultiplier,
                                                            float temperatureMultiplier)
  {
    Debug.Assert(cpuctMultiplier >= 0);

    GatheredChildStats stats = CheckInitThreadStatics();

    Debug.Assert(numTargetVisits >= 0);
    Debug.Assert(minChildIndex == 0); // implementation restriction
    Debug.Assert(maxChildIndex <= PUCTScoreCalcVector.MAX_CHILDREN);
    Debug.Assert(node.IsLocked);

    ref readonly GNodeStruct nodeRef = ref node.NodeRef;

    int numToProcess = Math.Min(Math.Min(maxChildIndex + 1, nodeRef.NumPolicyMoves), 
                                PUCTScoreCalcVector.MAX_CHILDREN);

    if (numToProcess == 0)
    {
      return new NodeSelectAccumulator(int.MinValue, double.NaN, double.NaN, 0);
    }

    // Gather necessary fields
    // TODO: often NInFlight of parent is null (thus also children) and we could
    //       have special version of Gather which didn't bother with that


    graph.GatherChildInfoViaChildren(node, selectorID, maxChildIndex, dualCollisionFraction, stats, refreshStaleEdges);

    double[] qWhenNoChildrenComposite = null;
    if (MCGSParamsFixed.EXPERIMENTAL_FPU_VIA_RPO_ENABLED && numToProcess > 1 && node.NumEdgesVisited > 0)
    {
      qWhenNoChildrenComposite = ImputeQForUnvisitedChildren(node, stats);
    }

    // Possibly apply supplemental temperature scaling.
    if (temperatureMultiplier != 1 && numToProcess > 1)
    {
      TemperatureScaler.ApplyTemperature(node.NumPolicyMoves, stats.P.Span[..numToProcess], 
                                         stats.SumPVisited, temperatureMultiplier);
    }


    if (false && paramsSearch.TestFlag)
    {
      Span<double> uncertaintyPolicySpan = stats.UP.Span;
      Span<double> uncertaintyValueSpan = stats.UV.Span;
      Span<double> nSpan = stats.N.Span;
      Span<double> wSpan = stats.W.Span;
      for (int i = 0; i < Math.Min(node.NumEdgesVisited, numToProcess); i++)
      {
        const double VALUE_UNCERTAINTY_WEIGHT = 1.0f;
        double n = Math.Max(1, nSpan[i]);
        double adjust = (VALUE_UNCERTAINTY_WEIGHT * uncertaintyValueSpan[i]) / Math.Sqrt(n + 1);
        wSpan[i] -= adjust * nSpan[i];
      }
    }


    // In old class MCTSNodeSTructScoreCalc see implementations of:
    //      if (Context.ParamsSelect.PolicyDecayFactor > 0)
    // Katago CPUCT scaling technique (TestFlag2)


    // Possibly disqualify pruned moves from selection.
    if (node.IsSearchRoot && rootMovePruningStatus != null
   && numTargetVisits != 0) // do not skip any if only querying all scores          
    {
      Span<double> gatherStatsNSpan = stats.N.Span;
      Span<double> gatherStatsWSpan = stats.W.Span;
      for (int i = 0; i < numToProcess; i++)
      {
        // Note that moves are never pruned if the do not yet have any visits
        // because otherwise the subsequent leaf selection will never 
        // be able to proceed beyond this unvisited child.
        if (rootMovePruningStatus[i] != Managers.MCGSFutilityPruningStatus.NotPruned
         && gatherStatsNSpan[i] > 0)
        {
          // At root the search wants best Q values 
          // but because of minimax prefers moves with worse Q and W for the children
          // Therefore we set W of the child very high to make it discourage visits to it.
          gatherStatsWSpan[i] = double.MaxValue;
        }
      }
    }

    // If any child is a checkmate then exploration is not appropriate,
    // set cpuctMultiplier to low value as an elegant means of effecting certainty propagation
    // (no changes to algorithm are needed, all subsequent visits will go to this terminal node).
    if (ParamsSearch.CheckmateCertaintyPropagationEnabled && nodeRef.CheckmateKnownToExistAmongChildren)
    {
      const bool ALLOW_MINIMAL_EXPORATION = true;
      if (ALLOW_MINIMAL_EXPORATION)
      {
        // Minimal exploration may allow "better mates" to be eventually found
        // (e.g. a tablebase mate in 3 instead of mate in 30).
        cpuctMultiplier = 0.1f;
      }
      else
      {
        cpuctMultiplier = 0f;
        numToProcess = Math.Min(numToProcess, node.NumEdgesExpanded);
      }
    }

    double sumPVisited = stats.SumPVisited;

#if DEBUG
    double sumPVisitedRecalc = 0;
    for (int i=0;i<node.NumEdgesVisited;i++)
    {
      GEdge childEdge = node.ChildEdgeAtIndex(i);
      if (childEdge.N > 0)
      {
        // Debug.Assert(childEdge.N > 0); not true if parallel enabled
        sumPVisitedRecalc += childEdge.P;
      }
    }
    // Non-agreement here due to NumChildrenVisited not yet counting nodes with N=0, but NInFlight>0
    //Debug.Assert(Math.Abs(sumPVisited - sumPVisitedRecalc) < 1e-6);
    // Therefore do this weaker test:
    Debug.Assert(sumPVisited >= sumPVisitedRecalc);
#endif

    int numVisitsAccepted = 0;
    if (numToProcess == 1 && scores.IsEmpty)
    {
      // No need to compute in this special case of only child to consider and scores not requested.
      childVisitCounts[0] = (short)numTargetVisits;
      numVisitsAccepted = numTargetVisits;
    }
    else
    {
      // previously: int parentNumInFlightX = selectorID == 0 ? nodeRef.NInFlight : nodeRef.NInFlight1;      
      // TODO: Tests at 50 and 500 nodes/move suggest setting this always zero is better?
      //       Probably this is not correct, reflects only a poor tuning of CPUCT,
      //       and this is just having effect of backdoor CPUCT change.
      double parentNumInFlight = stats.SumNumInFlightAll;


      // Compute scores of top children
      float thresholdPUCTSuboptimalityReject = float.MaxValue;
      if (paramsSearch.VisitSuboptimalityRejectThreshold != null)
      {
        thresholdPUCTSuboptimalityReject = paramsSearch.VisitSuboptimalityRejectThreshold.Value;
      }

      numVisitsAccepted = PUCTScoreCalcVector.ScoreCalcMulti(paramsSelect,
                                                              node.IsSearchRoot, nodeRef.N,
                                                              parentNumInFlight,
                                                              nodeRef.Q, sumPVisited,
                                                              stats,
                                                              qWhenNoChildrenComposite,
                                                              numToProcess, numTargetVisits,
                                                              scores, childVisitCounts, cpuctMultiplier,
                                                              paramsSearch.ActionHeadSelectionWeight,
                                                              thresholdPUCTSuboptimalityReject);      

      if (numTargetVisits > 0 && MCGSParamsFixed.OUT_OF_ORDER_CHILDREN_ALLOWED)
      {
        FillInSequentialVisitHoles(childVisitCounts, ref node.NodeRef, numToProcess);
      }
    }

    // Return accumulated value across all children and also contribution from the node itself.
    double nToUse = node.Terminal.IsTerminal() ? node.N : 1;
    return new NodeSelectAccumulator(nToUse + gatherStats.SumNVisited,
                                     (nToUse * (double)nodeRef.V) + -gatherStats.SumWVisited,
                                     (nToUse * (double)nodeRef.DrawP) + gatherStats.SumDVisited,
                                     numVisitsAccepted);
  }

  private static double[] ImputeQForUnvisitedChildren(GNode node, GatheredChildStats stats)
  {
    double[] qWhenNoChildrenComposite;
    double[] pi = new double[stats.P.Length];
    double sumPi = 0;
    for (int i = 0; i < pi.Length; i++)
    {
      double thisPi = stats.P.Span[i];
      sumPi += thisPi;
      pi[i] = thisPi;
    }

    // normalize
    for (int i = 0; i < pi.Length; i++)
    {
      pi[i] /= sumPi;
    }

    int bestIndex = 0;
    double bestQ = float.MinValue;
    for (int i = 0; i < node.NumEdgesVisited; i++)
    {
      double thisQ = stats.W.Span[i] / stats.N.Span[i];
      if (thisQ > bestQ)
      {
        bestIndex = i;
        bestQ = thisQ;
      }
    }

    static double[] Avg(Span<double> p0, Span<double> p1)
    {
      double[] ret = new double[p1.Length];
      for (int i = 0; i < p1.Length; i++)
      {
        ret[i] = 0.5f * p0[i] + 0.5f * p1[i];
      }
      return ret;
    }

    const double TAU = 0.10;// when fitted from sample data from small/medium-sized graphs we say 0.07 (low N) to 0.13 (higher N)

    void SetQFromChild(int childIndex, Span<double> ret)
    {
      double q = stats.W.Span[childIndex] / stats.N.Span[childIndex];
      BoltzmannValueCalibrator.ComputeQFromPolicy_AnchorChild(pi, childIndex, -q, TAU, ret,
                                                              renormalizeIfNeeded: false, clipMin: -1.2, clipMax: 1.2);
    }

    // Using parent as anchor didn't seem to work as well
    //      double[] qWhenNoChildrenPerChildParent = new double[stats.P.Length];
    //      BoltzmannValueCalibrator.ComputeQFromPolicy_MatchParentValue(pi, node.V, TAU, qWhenNoChildrenPerChildParent);

    Span<double> qWhenNoChildrenPerBestChild = stackalloc double[stats.P.Length];
    SetQFromChild(bestIndex, qWhenNoChildrenPerBestChild);

    Span<double> qWhenNoChildrenPerClosestChild = stackalloc double[stats.P.Length];
    SetQFromChild(node.NumEdgesVisited - 1, qWhenNoChildrenPerClosestChild);

    qWhenNoChildrenComposite = Avg(qWhenNoChildrenPerBestChild, qWhenNoChildrenPerClosestChild);
    return qWhenNoChildrenComposite;
  }


  /// <summary>
  /// Ceres algorithms require children to be visited strictly sequentially,
  /// so no child is visited before all of its siblings with smaller indices have already been visited.
  /// 
  /// This method insures this condition is always satisfied by shifting leftward
  /// any children which otherwise be to the right of some unexpanded node.
  /// </summary>
  /// <param name="childVisitCounts"></param>
  /// <param name="nodeRef"></param>
  /// <param name="numToProcess"></param>
  private static void FillInSequentialVisitHoles(Span<short> childVisitCounts, 
                                                 ref readonly GNodeStruct nodeRef, 
                                                 int numToProcess)
  {
    // Fixup any holes
    int numExpanded = nodeRef.NumEdgesExpanded;
    for (int i = numExpanded; i < numToProcess; i++)
    {
      if (childVisitCounts[i] == 0)
      {
        for (int j = numToProcess - 1; j > i; j--)
        {
          if (childVisitCounts[j] > 0)
          {
            childVisitCounts[i] = 1;
            childVisitCounts[j]--;
            break;
          }
        }
      }
    }
  }


  /// <summary>
  /// Computes the UCT scores (used to select best child) for all children
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="node"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="selectorID"></param>
  /// <param name="dualCollisionFraction"></param>
  /// <param name="cpuctMultiplier"></param>
  /// <param name="temperatureMultiplier"></param>
  /// <returns></returns>
  public static double[] CalcChildScores(Graph graph, 
                                         GNode node,
                                         ParamsSearch paramsSearch, 
                                         ParamsSelect paramsSelect,
                                         int selectorID, 
                                         bool refreshStaleEdges,
                                         float dualCollisionFraction = 0.25f, 
                                         float cpuctMultiplier = 1, 
                                         float temperatureMultiplier = 1)
  {
    double[] scores = new double[node.NodeRef.NumPolicyMoves];

    ComputeTopChildScores(graph, node, paramsSearch, paramsSelect, selectorID, refreshStaleEdges, null,
                          dualCollisionFraction, 0, node.NodeRef.NumPolicyMoves - 1,
                          1, scores, default, cpuctMultiplier, temperatureMultiplier);
    return scores;
  }

}
