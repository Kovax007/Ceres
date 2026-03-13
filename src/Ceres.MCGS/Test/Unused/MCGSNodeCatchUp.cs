#if NOT_USED
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
using System.Runtime.CompilerServices;

using Ceres.Base.Math;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Iteration.Params;
using Ceres.MCGS.Storage;
using Ceres.MCGS.Storage.Structs;
using SharpCompress;

#endregion

namespace Ceres.MCGS.Search;

public static class MCGSNodeCatchUp
{
//  static int COUNT = 0;
  public static (int numDirty, int numUpdated) AllNodesSelfUpdate(MCGSEngine coordinator)
  {
    // Only every other time
//    if (COUNT++ % 3 == 1) return;

    int numUpdated = 0;
    int numDirty = 0;
    // TOOD: make this more convenient to use

    NodeUtils.VisitSubtreeBreadthFirst(coordinator.Graph, coordinator.Manager.Engine.SearchRootNode.CalcPosition(),
              new NodeIndex(GraphStore.ROOT_NODE_INDEX),
              (GNode node, MGPosition mgPos) =>
              {
                if (node.NodeRef.miscFields.IsDirty)
                {
                  numDirty++;
                  bool updated = NodeSelfUpdate(coordinator, node);
                  if (updated)
                  {
                    numUpdated++;
                    //  Console.WriteLine(found + "  " + node);
                  }
                  node.NodeRef.miscFields.IsDirty = false;
                }
                // Console.WriteLine("Tree visit " + node + " hash=" + MGPositionHashing.HashValue96(in mgPos, MCGSParamsFixed.HASH_MODE).Low % 10_000 + " " + mgPos.ToPosition.FEN + " ");
                return true;
              });
    //    Console.WriteLine(numFound + " of " + Graph.RootNode.N);
    return (numDirty, numUpdated);
  }


  public static bool NodeSelfUpdate(MCGSEngine engine, GNode node)
  {
    Span<int> edgeN = stackalloc int[node.NumEdgesExpanded];
    Span<int> childN = stackalloc int[node.NumEdgesExpanded];

    // Collect current visit counts; quit early if none of the children have excess child counts.
    if (!GatherEdgeAndChildCounts(node, edgeN, childN))
    {
      return false;
    }

    // Run PUCT algorithm to determine which child would get visit if we had one more visit to allocate.
    MCGSSelectBackupStrategyBase strategy = engine.Strategy;
    strategy.SelectChildren(node, 0, 1, node.NumEdgesExpanded, 1, false, 1.0f, 1.0f,
                            engine.Manager.RootMovesPruningStatus,
                            out Span<short> childVisitCounts, out Span<double> scores);

    // Look for one eligible child and apply the supplemental backup if found.
    for (int ix = 0; ix < node.NumEdgesExpanded; ix++)
    {
      bool shouldBackup = childVisitCounts[ix] > 0 && childN[ix] > edgeN[ix];
      if (shouldBackup)
      {
        const bool ALSO_OTHER_PARENTS = false;
        PerformBackup(engine, node, ix, ALSO_OTHER_PARENTS);
        return true;
      }
    }

    return false;
  }


  /// <summary>
  /// Gathers values of edge N and child N across all children.
  /// Returns if at least one child has child N in excess of edge N.
  /// </summary>
  private static bool GatherEdgeAndChildCounts(GNode node, Span<int> edgeN, Span<int> childN)
  {
    bool hasExcess = false;
    int i = 0;
    foreach (GEdge edge in node.ChildEdgesExpanded)
    {
      edgeN[i] = edge.N;
      childN[i] = edge.ChildNode.N;
      hasExcess |= childN[i] > edgeN[i];
      i++;
    }
    return hasExcess;
  }


  public static double ComputeCatchUpValue(int edgeN, double childQ1, double childQ2)
        => childQ1 + (edgeN + 1) * (childQ2 - childQ1);

  /// <summary>
  /// Calculate a synthetic value V★ that, when applied mSamples times,
  /// reconciles the edge’s running average with the child’s new average.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static double ComputeCatchUpValue(int edgeN, double edgeQ, double childQ, int nSamples)
  {
    if (nSamples <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(nSamples), "Must be >= 1");
    }

    //  V* = [(N + m)·Q_child – N·Q_edge] / n
    return ((edgeN + nSamples) * childQ - edgeN * edgeQ) / nSamples;
  }

  /// <summary>
  /// Executes the strategy-specific backup for the selected child.
  /// </summary>
  private static void PerformBackup(MCGSEngine engine, GNode node, int childIndex, bool alsoUpdateOtherParents)
  {
    MCGSSelectBackupStrategyBase strategy = engine.Strategy;
    GEdge edge = node.ChildEdgeAtIndex(childIndex);

    throw new NotImplementedException();
#if NOT
    // Forward backup (child -> edge -> node)
    strategy.BackupToEdge(edge, true, 1, double.NaN, double.NaN, false); // B 

    double catchUpV = ComputeCatchUpValue(edge.N, edge.Q, edge.ChildNode.Q);

    // 50% blend of catch-up value and child value
    // extensive tests at 30,000 nodes/move with T81
    // showed this is about Elo equal with no catch-up (-3 Elo, +5 Elo)
    // (whereas using purely the catch-up value is -10 Elo)
    catchUpV = 0.5 * edge.ChildNode.Q
             + 0.5 * StatUtils.Bounded(catchUpV, -1, 1);

    //strategy.BackupToNode(node, 1, edge.ChildNode.V, (float)edge.ChildNode.D, false); // C
    throw new NotImplementedException("Need remediation next line for pseudoTransposition arguments");
    strategy.BackupToNode(node, 1, catchUpV, edge.ChildNode.D, false, engine.Manager.ParamsSearch.NodeRecalculationPhase == MCGSPhase.Backup);

    // Back-propagate to parents (if any)
    if (alsoUpdateOtherParents && !node.IsGraphRoot)
    {
//      foreach (GEdge parentEdge in node.ParentEdges)
      {
//        strategy.BackupToEdge(parentEdge, true, 0, -edge.ChildNode.V, (float)edge.ChildNode.D, true);
      }
    }
#endif
  }

}

#endif