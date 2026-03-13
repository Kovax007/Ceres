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
using Ceres.Chess.GameEngines;
using Ceres.Chess.Positions;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.EngineTests;

/// <summary>
/// Parameters which control CompareEnginesVersusOptimal runs.
/// </summary>
/// <param name="Description"></param>
/// <param name="PGNFileName"></param>
/// <param name="NumPositions"></param>
/// <param name="PosFilter"></param>
/// <param name="Player1Engine"></param>
/// <param name="Player2Engine"></param>
/// <param name="ArbiterEngine"></param>
/// <param name="Limit"></param>
/// <param name="GPUIDs"></param>
/// <param name="ParamsSearch1"></param>
/// <param name="ParamsSelect1"></param>
/// <param name="ParamsSearch2"></param>
/// <param name="ParamsSelect2"></param>
/// <param name="Verbose"></param>
/// <param name="Engine1LimitMultiplier"></param>
/// <param name="EngineArbiterLimitMultiplier"></param>
/// <param name="RunStockfishCrosscheck"></param>
/// <param name="PosResultCallback"></param>
/// <param name="QDiffThresholdDumpVerboseMoveStats"></param>
public record CompareEngineParams(string Description, string PGNFileName, int NumPositions,
                                  Predicate<PositionWithHistory> PosFilter,
                                  GameEngine Player1Engine,
                                  GameEngine Player2Engine,
                                  GameEngine ArbiterEngine,
                                  SearchLimit Limit, int[] GPUIDs = null,
                                  ParamsSearch ParamsSearch1 = null, ParamsSelect ParamsSelect1 = null,
                                  ParamsSearch ParamsSearch2 = null, ParamsSelect ParamsSelect2 = null,
                                  bool Verbose = true,
                                  float Engine1LimitMultiplier = 1.0f, float EngineArbiterLimitMultiplier = 7f,
                                  bool RunStockfishCrosscheck = false,
                                  Action<CompareEnginePosResult> PosResultCallback = null,
                                  float QDiffThresholdDumpVerboseMoveStats = float.MaxValue)
{
}
