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

#region Using directives

using System;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.GameEngines;

using Ceres.Features.GameEngines;
using Ceres.MCGS.Iteration.Params;
using Ceres.Chess.NNEvaluators;
using Ceres.MCGS.Graphs.Structs;
using Ceres.MCGS.MCTSNodes.NEW.Test;
using Ceres.MCGS.MCTSNodes.NEW;

#endregion

namespace Ceres.MCGS.GameEngines
{
  /// <summary>
  /// Game definition for an MCGS engine.
  /// </summary>
  [Serializable]
  public class GameEngineDefCeresMCGS : GameEngineDef
  {
    public override bool SupportsNodesPerGameMode => false;

    public readonly NNEvaluatorDef EvaluatorDef;

    ParamsSearch SearchParams;
    ParamsSelect SelectParams;
    public readonly bool DisposeGraphAfterSearch;

    //    public static Func<(GameEngineCeresMCGSInProcess engine, PositionWithHistory Pos, SearchLimit Limit), (MGMove, float, int)> MoveMaker;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="evaluatorDef"></param>
    /// <exception cref="ArgumentNullException"></exception>
    public GameEngineDefCeresMCGS(string id, NNEvaluatorDef evaluatorDef,
                                  ParamsSearch searchParams,
                                  ParamsSelect selectParams,
                                  bool disposeGraphAfterSearch = true) : base(id)
    {
      if (evaluatorDef == null)
      {
        throw new ArgumentNullException(nameof(evaluatorDef));
      }

      EvaluatorDef = evaluatorDef;

      SearchParams = searchParams;
      SelectParams = selectParams;
      DisposeGraphAfterSearch = disposeGraphAfterSearch;

    }


    public override GameEngine CreateEngine()
    {
      GameEngineCeresMCGSInProcess ret = new(ID, EvaluatorDef, null,
                                              SearchParams, SelectParams, disposeGraphAfterSearch: DisposeGraphAfterSearch);


      ret.Warmup();

      return ret;
    }


    public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
    {
      if (deviceIndexIncrement != 0)
      {
        throw new NotImplementedException();
      }
    }

  }
}
#endif