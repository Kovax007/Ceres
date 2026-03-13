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
using Ceres.Base.OperatingSystem;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Features.GameEngines;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Test;

/// <summary>
/// Enum representing available engines for centralized creation.
/// </summary>
public enum EngineType
{
  CeresMCGS,
  CeresMCGS2,
  CeresMCTS,
  LC0_DAG,
  LC0_DAG_CUDA,
  LC0_Classic,
  CeresProd
}


/// <summary>
/// Enum representing LC0 engine types.
/// </summary>
public enum LC0EngineType
{
  Legacy,
  RewriteClassic,
  RewriteDAG,
  RewriteDAG_CUDA,
  RewriteDAGPrior,
  TCEC_DAG
}


/// <summary>
/// Static class providing centralized creation of game engines.
/// </summary>
public static class GameEnginesCommon
{
  /// <summary>
  /// Creates a game engine of the specified type.
  /// </summary>
  /// <param name="engineType">Type of engine to create</param>
  /// <param name="netCeres">Neural network specification for Ceres engines</param>
  /// <param name="netLC0">Neural network specification for LC0 engines</param>
  /// <param name="device">Device specification (e.g., "GPU:0#TensorRT16")</param>
  /// <returns>The created game engine</returns>
  public static GameEngine CreateGameEngine(EngineType engineType,
                                            string netCeres,
                                            string netLC0,
                                            string device = "GPU:0#TensorRT16")
  {
    NNEvaluatorDef evaluatorDefCeres = netCeres == null ? null : NNEvaluatorDef.FromSpecification(netCeres, device);
    NNEvaluatorDef evaluatorDefLc0 = netLC0 == null ? null : NNEvaluatorDef.FromSpecification(netLC0, device);

    return engineType switch
    {
      EngineType.CeresMCGS => CreateCeresMCGSEngine(evaluatorDefCeres),
      EngineType.CeresMCGS2 => CreateCeresMCGS2Engine(evaluatorDefCeres),
      EngineType.CeresMCTS => CreateCeresMCTSEngine(evaluatorDefCeres),
      EngineType.LC0_DAG => CreateLC0Engine(netLC0, "GPU:0#TensorRT", LC0EngineType.RewriteDAG, MCGSTest.SEARCH_PARAMS_MCGS, MCGSTest.SELECT_PARAMS_MCGS, verboseMoveOutput: true),
      EngineType.LC0_DAG_CUDA => CreateLC0Engine(netLC0, "GPU:0#TensorRT", LC0EngineType.RewriteDAG_CUDA, MCGSTest.SEARCH_PARAMS_MCGS, MCGSTest.SELECT_PARAMS_MCGS, verboseMoveOutput: true),
      EngineType.LC0_Classic => CreateLC0Engine(netLC0, "GPU:0#TensorRT", LC0EngineType.RewriteClassic, MCGSTest.SEARCH_PARAMS_MCGS, MCGSTest.SELECT_PARAMS_MCGS, verboseMoveOutput: true),
      EngineType.CeresProd => CreateCeresProductionEngine(netCeres),
      _ => throw new ArgumentException($"Unsupported engine type: {engineType}", nameof(engineType))
    };
  }


  /// <summary>
  /// Creates a Ceres MCGS engine.
  /// </summary>
  private static GameEngineCeresMCGSInProcess CreateCeresMCGSEngine(NNEvaluatorDef evaluatorDef)
  {
    return new GameEngineCeresMCGSInProcess("Ceres v2 MCGS", evaluatorDef,
                                            searchParams: MCGSTest.SEARCH_PARAMS_MCGS with { FutilityPruningStopSearchEnabled = false },
                                            selectParams: MCGSTest.SELECT_PARAMS_MCGS,
                                            disposeGraphAfterSearch: false);
  }


  /// <summary>
  /// Creates a Ceres MCGS2 engine.
  /// </summary>
  private static GameEngineCeresMCGSInProcess CreateCeresMCGS2Engine(NNEvaluatorDef evaluatorDef)
  {
    return new GameEngineCeresMCGSInProcess("Ceres v2 MCGS2", evaluatorDef,
                                            searchParams: MCGSTest.SEARCH_PARAMS_MCGS2 with { FutilityPruningStopSearchEnabled = false },
                                            selectParams: MCGSTest.SELECT_PARAMS_MCGS2,
                                            disposeGraphAfterSearch: false);
  }


  /// <summary>
  /// Creates a Ceres MCTS Classic engine.
  /// </summary>
  private static GameEngineCeresInProcess CreateCeresMCTSEngine(NNEvaluatorDef evaluatorDef)
  {
    return new GameEngineCeresInProcess("Ceres v2 MCTS", evaluatorDef,
                                        searchParams: MCGSTest.SEARCH_PARAMS_MCTS(MCGSTest.SEARCH_PARAMS_MCGS) with { FutilityPruningStopSearchEnabled = false },
                                        childSelectParams: new Ceres.MCTS.Params.ParamsSelect());
  }


  /// <summary>
  /// Creates a Ceres production engine (via UCI).
  /// </summary>
  private static GameEngine CreateCeresProductionEngine(string netCeres)
  {
    return CreateCeresUCIEngineDef(netCeres).CreateEngine();
  }


  /// <summary>
  /// Creates a Ceres UCI engine definition.
  /// </summary>
  public static GameEngineDefCeresUCI CreateCeresUCIEngineDef(string testNet)
  {
    static string exeCeres() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net8.0/Ceres.dll"
                                                        : @"C:\dev\ceres\artifacts\release\net8.0\ceres.exe";
    NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(testNet, "GPU:0");
    GameEngineDefCeresUCI engineDefCeresUCI = new("CeresUCINew", evalDef, overrideEXE: exeCeres(),
                                                  disableFutilityStopSearch: !MCGSTest.SEARCH_PARAMS_MCGS_COMMON.FutilityPruningStopSearchEnabled,
                                                  paramsSearch: MCGSTest.SEARCH_PARAMS_MCTS(MCGSTest.SEARCH_PARAMS_MCGS),
                                                  paramsSelect: new Ceres.MCTS.Params.ParamsSelect());
    return engineDefCeresUCI;
  }


  /// <summary>
  /// Creates an LC0 engine.
  /// </summary>
  public static GameEngine CreateLC0Engine(string testNet, string device, LC0EngineType type,
                                           ParamsSearch paramsSearch, ParamsSelect paramsSelect,
                                           bool disableFutilityPruning = false, bool verboseMoveOutput = false,
                                           string extraUCIOptions = null)
  {
    return CreateLC0EngineDef(testNet, device, type, paramsSearch, paramsSelect,
                              disableFutilityPruning, verboseMoveOutput, extraUCIOptions).CreateEngine();
  }


  /// <summary>
  /// Creates an LC0 engine definition.
  /// </summary>
  public static GameEngineDef CreateLC0EngineDef(string testNet, string device, LC0EngineType type,
                                                 ParamsSearch paramsSearch, ParamsSelect paramsSelect,
                                                 bool disableFutilityPruning = false, bool verboseMoveOutput = false,
                                                 string extraUCIOptions = null)
  {
    (string EXE, string extraArgs) lc0Info = type switch
    {
      LC0EngineType.Legacy => (null, ""),
      LC0EngineType.RewriteClassic =>
        SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "")
                                : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", ""),

      LC0EngineType.RewriteDAG =>
        SoftwareManager.IsLinux ? ("/home/david/dev/lc0_321/lc0/build/release/lc0", "dag-preview")
                                : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", "dag-preview"),

      LC0EngineType.RewriteDAG_CUDA
        => SoftwareManager.IsLinux ? ("/home/david/dev/lc0_menkib/lc0/build/release/lc0-cuda", "dag-preview")
                                   : (@"C:\apps\lc0_swiss9\lc0\build\lc0-cuda.exe", "dag-preview"),

      LC0EngineType.RewriteDAGPrior =>
        SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "dag-preview")
                                : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", "dag-preview"),

      LC0EngineType.TCEC_DAG =>
        SoftwareManager.IsLinux ? ("/home/david/dev/LC0_DAG_TCEC/lc0/build/release/lc0", "")
                                : throw new NotImplementedException(),
      _ => throw new NotImplementedException()
    };

    NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(testNet, device);
    GameEngineDefLC0 engineDefLC0 = new(type.ToString(), evalDef,
                                        disableFutilityPruning || !MCGSTest.SEARCH_PARAMS_MCGS_COMMON.FutilityPruningStopSearchEnabled,
                                        MCGSTest.SEARCH_PARAMS_MCTS(paramsSearch), MCGSTest.SELECT_PARAMS_MCTS(paramsSelect),
                                        overrideEXE: lc0Info.EXE,
                                        alwaysFillInHistory: false,
                                        verbose: verboseMoveOutput,
                                        extraCommandLineArgs: lc0Info.extraArgs + " " + extraUCIOptions,
                                        disableTreeReuse: !MCGSTest.SEARCH_PARAMS_MCGS_COMMON.GraphReuseEnabled);
    return engineDefLC0;
  }
}
