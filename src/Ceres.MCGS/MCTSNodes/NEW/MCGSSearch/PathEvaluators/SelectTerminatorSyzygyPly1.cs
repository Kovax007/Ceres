#if NOT
#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using Ceres.Base.Environment;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCGS.Search.PathEvaluators
{
  /// <summary>
  /// Selection terminator which can evaluate positions just 1 ply short
  /// of being covered by a tablebase. Succeeds only in the special situation where:
  ///   - the number of pieces on board is exactly one more than our tablebases cover, and
  ///   - there exists at least one capture move which leads to a tablebase loss for the opponent
  ///   
  /// Empirically the number of successful evaluations
  /// of SelectTerminatorSyzygyPly1 is typically approximately 10% that 
  /// of SelectTerminatorSyzygyPly successful evaluations.
  /// </summary>
  public sealed class SelectTerminatorSyzygyPly1 : SelectTerminatorBase
  {
    /// <summary>
    /// Supporting tablebase evaluator.
    /// </summary>
    public readonly SelectTerminatorSyzygy Ply0Evaluator;

    /// <summary>
    /// Number of probe successes.
    /// </summary>
    internal static AccumulatorMultithreaded NumHits;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="ply0Evaluator"></param>
    public SelectTerminatorSyzygyPly1(SelectTerminatorSyzygy ply0Evaluator, bool forceNoTablebaseTerminals)
    {
      Ply0Evaluator = ply0Evaluator;
      if (forceNoTablebaseTerminals)
      {
        Mode = SelectTerminatorMode.SetAuxilliaryEval;
      }
    }


    /// <summary>
    /// Implementation of evaluation method.
    /// </summary>
    /// <param name="path"></param>
    /// <returns></returns>
    protected override bool DoTryTerminate(MCGSPath path, ref SelectTerminationInfo terminationInfo)
    {
      if (path.NumVisitsInPath == 1)
      {
        // Don't attempt at the root to avoid short-circuiting search here
        // (we need to build a tree to be able to choose a best move).
        return false;
      }

      ref readonly MCGSPathVisit leafRef = ref path.LastVisitRef;

      // Abort immediately unless this position has exactly
      // one more piece than the max cardinality of our tablebases.
      if (path.LeafPositionAndMovesRef.Position.PieceCount != (Ply0Evaluator.MaxCardinality + 1))
      { 
        return false;
      }

      // Iterate over the capture moves.
      foreach ((MGMove move, MGPosition newPos) in PositionsGenerator1Ply.GenPositions(path.LeafPositionAndMovesRef.Position, move => move.Capture))
      {
        // Check if this position is in tablebase and it is a definitive win for our side.
        SelectTerminationInfo ply0TerminationInfo = default;
        bool ply0Found = Ply0Evaluator.Lookup(path, in newPos, ref ply0TerminationInfo);
        if (ply0Found && ply0TerminationInfo.TerminalStatus == GameResult.Checkmate)
        {
          // Check if loss for them (win for us)
          bool posLoses = ply0TerminationInfo.V < 0;
          if (!posLoses)
          {
            return false;
          }

          if (CeresEnvironment.MONITORING_METRICS)
          {
            NumHits.Add(1, (int)path.LeafPositionAndMovesRef.Position.A);
          }

          throw new System.Exception("Verify the values below. Does Checkmate always refer to loss when used elsewhere? Is SideToMove below correct?");
          terminationInfo = new (/*SelectTerminatorSyzygyPly1*/
                                 path.LeafPositionAndMovesRef.Position.SideToMove, MCGSPathTerminationReason.LeafTerminal, GameResult.Checkmate,
                                 ply0TerminationInfo.LossP, ply0TerminationInfo.WinP, ply0TerminationInfo.M,
                                 ply0TerminationInfo.UncertaintyV, ply0TerminationInfo.UncertaintyP);
          return true;
        }
      }

      return false;
    }


    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHits.Initialize();
    }
  }
}

#endif