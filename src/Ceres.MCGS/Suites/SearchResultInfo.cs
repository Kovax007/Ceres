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

using Ceres.Chess.MoveGen;
using Ceres.MCGS.Search;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using System;

#endregion

namespace Ceres.MCGS.Features.Suites
{
  public class SearchResultInfo
  {
    public readonly double Q;
    public readonly string UCIInfoString;
    public readonly MGMove BestMove;
    public readonly int N;
    public readonly int NumNodesWhenChoseTopNNode;
    public readonly int NumNNBatches;
    public readonly int NumNNNodes;
    public readonly int TopNNodeN;
    public readonly float FractionNumNodesWhenChoseTopNNode;
    public readonly float AvgDepth;
    public readonly float MaxDepth;
    public readonly float MAvg;
    public readonly float NodeSelectionYieldFrac;
    public readonly string PickedNonTopNMoveStr;

    public SearchResultInfo(MCGSManager manager, BestMoveInfoMCGS bestMove)
    {
      Q = manager.Engine.SearchRootNode.Q;
      //UCIInfoString = manager.UCIInfoString();
      // SearchPrincipalVariation pv1 = new SearchPrincipalVariation(worker1.Root);
      BestMove = bestMove.BestMove;
      N = manager.Engine.SearchRootNode.N;
      NumNodesWhenChoseTopNNode = manager.NumNodesWhenChoseTopNNode;
      NumNNBatches =  0;// manager.Context.NumNNBatches;
      NumNNNodes = 0;//manager.Context.NumNNNodes;

      TopNNodeN = 0;//manager.TopNChildIndex is null ? 0 : manager.TopNChildN;
      FractionNumNodesWhenChoseTopNNode = 0;//manager.FractionNumNodesWhenChoseTopNNode;
      AvgDepth = manager.AvgDepth;
      MaxDepth = manager.MaxDepth;
      MAvg = 0;//manager.Context.Root.MAvg;
      NodeSelectionYieldFrac = 0;//manager.Context.NodeSelectionYieldFrac;

      PickedNonTopNMoveStr = bestMove.BestMoveWasTopN ? " " : "!";
    }
  }

}
