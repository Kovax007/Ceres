#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ceres.Base.CUDA;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.Textual;
using Ceres.Features.GameEngines;
using Ceres.MCGS.GameEngines;
using static Ceres.Base.CUDA.CUDAUtilizationMetrics;


#endregion

namespace Ceres.MCGS.Test
{
  /// <summary>
  /// Enum representing available engines for testing.
  /// </summary>
  [Flags]
  public enum TestEngines
  {
    None = 0,
    CeresMCGS = 1 << 0,
    CeresMCGS2 = 1 << 1,
    CeresMCTS = 1 << 2,
    LC0_DAG = 1 << 3,
    LC0_DAG_CUDA = 1 << 4,
    LC0_Classic = 1 << 5,
    CeresProd = 1 << 6,
    All = CeresMCGS | CeresMCGS2 | CeresMCTS | LC0_DAG | LC0_DAG_CUDA | LC0_Classic | CeresProd
  }



  /// <summary>
  /// Test result entry for position testing across multiple engines.
  /// </summary>
  public record TestPositionResult
  {
    /// <summary>
    /// Name of the engine that generated this result.
    /// </summary>
    public readonly string EngineName;

    /// <summary>
    /// Description of the test (e.g., "Starting position, 1 min")
    /// </summary>
    public readonly string Description;

    /// <summary>
    /// Position with history that was tested.
    /// </summary>
    public readonly PositionWithHistory Position;

    /// <summary>
    /// Best move to be made.
    /// </summary>
    public readonly string CorrectMove;

    /// <summary>
    /// Move chosen by the engine.
    /// </summary>
    public readonly string ChosenMove;

    /// <summary>
    /// Final evaluation score from the engine.
    /// </summary>
    public readonly float FinalEvaluation;

    /// <summary>
    /// Number of nodes searched.
    /// </summary>
    public readonly int NumNodes;

    /// <summary>
    /// Runtime in seconds.
    /// </summary>
    public readonly double SecondsRuntime;

    public readonly string UCIInfo;

    public readonly CUDAUtilizationMetrics CUDAUtilization;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="engineName">Name of the engine</param>
    /// <param name="position">Position with history tested</param>
    /// <param name="description">Descripion of test position</param>
    /// <param name="chosenMove">Move chosen by engine</param>
    /// <param name="finalEvaluation">Final evaluation score</param>
    /// <param name="numNodes">Number of nodes searched</param>
    /// <param name="secondsRuntime">Runtime in seconds</param>
    public TestPositionResult(string engineName, PositionWithHistory position, 
                              string description, string correctMove,
                              CUDAUtilizationMetrics cudaUtilization,
                              string chosenMove, 
                              float finalEvaluation, int numNodes, double secondsRuntime,
                              string uciInfo, EPDEntry epd = null)
    {
      EngineName = engineName;
      Position = position;
      Description = description;
      CorrectMove = correctMove;
      CUDAUtilization = cudaUtilization;
      ChosenMove = chosenMove;
      FinalEvaluation = finalEvaluation;
      NumNodes = numNodes;
      SecondsRuntime = secondsRuntime;
      UCIInfo = uciInfo;
      if (epd != null && epd.BMMoves != null && epd.BMMoves.Length > 0)
      {
        Move correct = Move.FromUCI(position.FinalPosition, chosenMove);
        CorrectMove = SANParser.FromSAN(epd.BMMoves[0], position.FinalPosition).Move.ToString().ToLower();

        // TODO: support all possible BM moves if mulitple?
//        int valueOfMove = epd.ValueOfMove(correct, position.FinalPosition);
      }
    }
}

  /// <summary>
  /// Class for testing positions across multiple chess engines and maintaining results.
  /// </summary>
  public class TestPositionMultipleEngines : IDisposable
  {
    public GameEngineCeresMCGSInProcess ceresEngineNEW;
    public GameEngineCeresMCGSInProcess ceresEngineNEW2;
    public GameEngineCeresInProcess ceresEngineClassic;
    public GameEngine lc0EngineDAG;
    public GameEngine lc0EngineDAGPrior;
    public GameEngine lc0EngineClassic;
    private GameEngine ceresEngineProduction;


    /// <summary>
    /// Public list of test results accessible at any time.
    /// </summary>
    public readonly List<TestPositionResult> Results = [];


    readonly string netCeres;
    readonly string netLC0;
    readonly NNEvaluatorDef evaluatorDefCeres;
    readonly NNEvaluatorDef evaluatorDefLc0;

    /// <summary>
    /// Constructor that creates the GameEngine objects up front.
    /// </summary>
    /// <param name="net">Neural network specification (e.g., "~T74")</param>
    public TestPositionMultipleEngines(string netCeres, string netLC0)
    {
      this.netCeres = netCeres;
      this.netLC0 = netLC0;
      evaluatorDefCeres = NNEvaluatorDef.FromSpecification(netCeres, "GPU:0#TensorRT16");
      evaluatorDefLc0 = netLC0 == null ? null : NNEvaluatorDef.FromSpecification(netLC0, "GPU:0#TensorRT16");
    }

    bool haveInitialized = false;


    public void Init(TestEngines enginesToUse)
    { 
      if (haveInitialized)
      {
        return;
      }
      haveInitialized = true;

      // Create Ceres MCGS NEW engine
      ceresEngineNEW = new GameEngineCeresMCGSInProcess("Ceres v2 MCGS", evaluatorDefCeres,
                                                           searchParams: MCGSTest.SEARCH_PARAMS_MCGS with { FutilityPruningStopSearchEnabled = false },
                                                           selectParams: MCGSTest.SELECT_PARAMS_MCGS,
                                                           disposeGraphAfterSearch: false);
      // Create Ceres MCGS NEW _TEST engine (identical settings, different name)
      ceresEngineNEW2 = new GameEngineCeresMCGSInProcess("Ceres v2 MCGS2", evaluatorDefCeres,
                                                           searchParams: MCGSTest.SEARCH_PARAMS_MCGS2 with { FutilityPruningStopSearchEnabled = false },
                                                           selectParams: MCGSTest.SELECT_PARAMS_MCGS2,
                                                           disposeGraphAfterSearch: false);

      // Create Ceres MCTS Classic engine
      if ((enginesToUse & TestEngines.CeresMCTS) != 0)
      {
        ceresEngineClassic = new GameEngineCeresInProcess("Ceres v2 MCTS", evaluatorDefCeres,
                                                        searchParams: MCGSTest.SEARCH_PARAMS_MCTS(MCGSTest.SEARCH_PARAMS_MCGS) with { FutilityPruningStopSearchEnabled = false },
                                                        childSelectParams: new Ceres.MCTS.Params.ParamsSelect());
      }

      // Create LC0 engines
      if ((enginesToUse & TestEngines.LC0_DAG) != 0)
      {
        lc0EngineDAG = MCGSTest.GameEngineLc0(netLC0, "GPU:0#TensorRT", MCGSTest.LC0EngineType.RewriteDAG, true).CreateEngine();
      }

      if ((enginesToUse & TestEngines.LC0_DAG_CUDA) != 0)
      {
        lc0EngineDAGPrior = MCGSTest.GameEngineLc0(netLC0, "GPU:0#TensorRT", MCGSTest.LC0EngineType.RewriteDAG_CUDA, true).CreateEngine();
      }

      if ((enginesToUse & TestEngines.LC0_Classic) != 0)
      {
        lc0EngineClassic = MCGSTest.GameEngineLc0(netLC0, "GPU:0#TensorRT", MCGSTest.LC0EngineType.RewriteClassic, true).CreateEngine();
      }

      // Create Ceres Production engine
      if ((enginesToUse & TestEngines.CeresProd) != 0)
      {
        ceresEngineProduction = GameEngineCeresUCI(netCeres).CreateEngine();
      }
    }


    /// <summary>
    /// Test a position with specified limit and description, on selected engines.
    /// </summary>
    /// <param name="testPosition">Position with history to test</param>
    /// <param name="limitToUse">Search limit to apply</param>
    /// <param name="enginesToUse">Engines to run the test on</param>
    /// <param name="testDescription">Description of the test</param>
    /// <param name="correctMove">Correct move for the position</param>
    /// <param name="epd">EPD entry for the position</param>
    /// <param name="verboseMode">Whether to enable verbose console output</param>
    public void TestPosition(PositionWithHistory testPosition, SearchLimit limitToUse,
                             TestEngines enginesToUse,
                             string testDescription, string correctMove, EPDEntry epd, 
                             bool verboseMode, bool validate)
    {
      if (verboseMode)
      {
        Console.WriteLine($"Testing position: {testDescription}");
        Console.WriteLine($"Position: {testPosition}");
        Console.WriteLine($"Search limit: {limitToUse}");
        Console.WriteLine();
      }

      Init(enginesToUse);

      // Test Ceres MCGS NEW
      if ((enginesToUse & TestEngines.CeresMCGS) != 0)
      {
        ceresEngineNEW.ResetGame();
        GameEngineSearchResultCeresMCGS resultMCGS;
        CUDAUtilizationMetrics cudaMCGS;
        using (new CUDAUtilizationBlock(out cudaMCGS))
        {
          resultMCGS = ceresEngineNEW.SearchCeres(testPosition, limitToUse);
        }

        Console.WriteLine("Ceres v2 MCGS " + resultMCGS.BestMoveInfo);
        if (verboseMode)
        {
          if (validate) resultMCGS.Engine.Graph.Validate();
          resultMCGS.Search.Manager.DumpFullInfo(resultMCGS, Console.Out, testDescription);
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Search with N={resultMCGS.FinalN} complete with Q {resultMCGS.ScoreQ}, graph validation passed.");
        }

        Results.Add(new TestPositionResult("Ceres v2 MCGS", testPosition, testDescription, correctMove, cudaMCGS, resultMCGS.MoveString,
                                           resultMCGS.ScoreQ, resultMCGS.FinalN, resultMCGS.TimingStats.ElapsedTimeSecs,
                                           ceresEngineNEW.UCIInfo?.RawString, epd));
      }

      // Test Ceres MCGS NEW2
      if ((enginesToUse & TestEngines.CeresMCGS2) != 0)
      {
        ceresEngineNEW2.ResetGame();
        GameEngineSearchResultCeresMCGS resultMCGS2;
        CUDAUtilizationMetrics cudaMCGS2 = default;
        using (new CUDAUtilizationBlock(out cudaMCGS2))
        {
          resultMCGS2 = ceresEngineNEW2.SearchCeres(testPosition, limitToUse);
        }
        Console.WriteLine("Ceres V2 MCGS2 " + resultMCGS2.BestMoveInfo);
        if (verboseMode)
        {
          if (validate) resultMCGS2.Engine.Graph.Validate();
          resultMCGS2.Search.Manager.DumpFullInfo(resultMCGS2, Console.Out, testDescription + "2");
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Search (TEST) with N={resultMCGS2.FinalN} complete with Q {resultMCGS2.ScoreQ}, graph validation passed.");
        }
        Results.Add(new TestPositionResult("Ceres v2 MCGS2", testPosition, testDescription, correctMove, cudaMCGS2, resultMCGS2.MoveString,
                                           resultMCGS2.ScoreQ, resultMCGS2.FinalN, resultMCGS2.TimingStats.ElapsedTimeSecs,
                                           ceresEngineNEW2.UCIInfo?.RawString, epd));
      }

      // Test Ceres MCTS Classic
      if ((enginesToUse & TestEngines.CeresMCTS) != 0)
      {
        if (verboseMode)
        {
          Console.WriteLine();
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "----------------- Ceres v1 inprocess -----------------");
        }
        ceresEngineClassic.ResetGame();
        GameEngineSearchResultCeres resultMCTS;
        CUDAUtilizationMetrics cudaMCTS;
        using (new CUDAUtilizationBlock(out cudaMCTS))
        {
          resultMCTS = ceresEngineClassic.SearchCeres(testPosition, limitToUse);
        }
        Console.WriteLine("Ceres v2 MCTS " + resultMCTS);
        if (verboseMode)
        {
          resultMCTS.Search.Manager.DumpFullInfo(Console.Out, testDescription);
        }

        Results.Add(new TestPositionResult("Ceres v2 MCTS", testPosition, testDescription, correctMove, cudaMCTS, resultMCTS.MoveString,
                                           (float)resultMCTS.ScoreQ, resultMCTS.FinalN, resultMCTS.TimingStats.ElapsedTimeSecs,
                                           ceresEngineClassic.UCIInfo?.RawString, epd));
      }

      // Test LC0 engines
      if ((enginesToUse & TestEngines.LC0_DAG) != 0)
      {
        GameEngine lc0Engine = lc0EngineDAG;
        string engineName = "LC0_DAG";

        if (verboseMode)
        {
          Console.WriteLine();
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"----------------- {engineName} -----------------");
        }

        lc0Engine.Warmup();
        lc0Engine.ResetGame();
        GameEngineSearchResult lc0Result;
        CUDAUtilizationMetrics cudaLC0;
        using (new CUDAUtilizationBlock(out cudaLC0))
        {
          lc0Result = lc0Engine.Search(testPosition, limitToUse, verbose: verboseMode);
        }
        Console.WriteLine(engineName + " " + lc0Result);
        if (verboseMode)
        {
          Console.WriteLine(lc0Result);
          foreach (VerboseMoveStat stat in lc0Result.VerboseMoveStats)
          {
            Console.WriteLine("  " + stat.ToString());
          }
        }

        Results.Add(new TestPositionResult(engineName, testPosition, testDescription, correctMove, cudaLC0, lc0Result.MoveString,
                                           lc0Result.ScoreQ, lc0Result.FinalN, lc0Result.TimingStats.ElapsedTimeSecs,
                                           lc0Engine.UCIInfo?.RawString, epd));
      }

      if ((enginesToUse & TestEngines.LC0_DAG_CUDA) != 0)
      {
        GameEngine lc0Engine = lc0EngineDAGPrior;
        string engineName = "LC0_DAG_CUDA";

        if (verboseMode)
        {
          Console.WriteLine();
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"----------------- {engineName} -----------------");
        }

        lc0Engine.Warmup();
        lc0Engine.ResetGame();
        GameEngineSearchResult lc0Result;
        CUDAUtilizationMetrics cudaLC0;
        using (new CUDAUtilizationBlock(out cudaLC0))
        {
          lc0Result = lc0Engine.Search(testPosition, limitToUse, verbose: verboseMode);
        }
        Console.WriteLine(engineName + " " + lc0Result);
        if (verboseMode)
        {
          Console.WriteLine(lc0Result);
          foreach (VerboseMoveStat stat in lc0Result.VerboseMoveStats)
          {
            Console.WriteLine("  " + stat.ToString());
          }
        }

        Results.Add(new TestPositionResult(engineName, testPosition, testDescription, correctMove, cudaLC0, lc0Result.MoveString,
                                           lc0Result.ScoreQ, lc0Result.FinalN, lc0Result.TimingStats.ElapsedTimeSecs,
                                           lc0Engine.UCIInfo?.RawString, epd));
      }

      if ((enginesToUse & TestEngines.LC0_Classic) != 0)
      {
        GameEngine lc0Engine = lc0EngineClassic;
        string engineName = "LC0_Classic";

        if (verboseMode)
        {
          Console.WriteLine();
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"----------------- {engineName} -----------------");
        }

        lc0Engine.Warmup();
        lc0Engine.ResetGame();
        GameEngineSearchResult lc0Result;
        CUDAUtilizationMetrics cudaLC0;
        using (new CUDAUtilizationBlock(out cudaLC0))
        {
          lc0Result = lc0Engine.Search(testPosition, limitToUse, verbose: verboseMode);
        }
        Console.WriteLine(engineName + " " + lc0Result);
        if (verboseMode)
        {
          Console.WriteLine(lc0Result);
          foreach (VerboseMoveStat stat in lc0Result.VerboseMoveStats)
          {
            Console.WriteLine("  " + stat.ToString());
          }
        }

        Results.Add(new TestPositionResult(engineName, testPosition, testDescription, correctMove, cudaLC0, lc0Result.MoveString,
                                           lc0Result.ScoreQ, lc0Result.FinalN, lc0Result.TimingStats.ElapsedTimeSecs,
                                           lc0Engine.UCIInfo?.RawString, epd));
      }

      // Test Ceres Production
      if ((enginesToUse & TestEngines.CeresProd) != 0)
      {
        if (verboseMode)
        {
          Console.WriteLine();
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"----------------- Ceres v1 Production -----------------");
        }
        GameEngineSearchResult ceresProductionResult;
        CUDAUtilizationMetrics cudaCeresProd = default;
        using (new CUDAUtilizationBlock(out cudaCeresProd))
        {
          ceresProductionResult = ceresEngineProduction.Search(testPosition, limitToUse, verbose: verboseMode);
        }
        Console.WriteLine("CeresProd " + ceresProductionResult);
        if (verboseMode)
        {
          Console.WriteLine(ceresProductionResult);
        }

        Results.Add(new TestPositionResult("CeresProd", testPosition, testDescription, correctMove, cudaCeresProd, ceresProductionResult.MoveString,
                                           ceresProductionResult.ScoreQ, ceresProductionResult.FinalN, ceresProductionResult.TimingStats.ElapsedTimeSecs,
                                           ceresEngineProduction.UCIInfo?.RawString, epd));
      }

      // Summary output (always shown even if verboseMode is false)
      Console.WriteLine("SUMMARY " + testDescription + " " + testPosition.FENAndMovesString);

      if ((enginesToUse & TestEngines.CeresProd) != 0)
      {
        TestPositionResult ceresProductionResult = Results.LastOrDefault(r => r.EngineName == "CeresProd");
        if (ceresProductionResult != null)
          Console.WriteLine($"CeresProd     {ceresProductionResult.ChosenMove,10}  {ceresProductionResult.FinalEvaluation,6:F2}   {ceresProductionResult.NumNodes,16:N0}  {ceresProductionResult.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.CeresMCTS) != 0)
      {
        var resultMCTS = Results.LastOrDefault(r => r.EngineName == "Ceres v2 MCTS");
        if (resultMCTS != null)
          Console.WriteLine($"CeresMCTS     {resultMCTS.ChosenMove,10}  {resultMCTS.FinalEvaluation,6:F2}  {resultMCTS.NumNodes,16:N0}   {resultMCTS.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.LC0_Classic) != 0)
      {
        var lc0ClassicResult = Results.LastOrDefault(r => r.EngineName == "LC0_Classic");
        if (lc0ClassicResult != null)
          Console.WriteLine($"LC0_Classic   {lc0ClassicResult.ChosenMove,10}  {lc0ClassicResult.FinalEvaluation,6:F2}  {lc0ClassicResult.NumNodes,16:N0}   {lc0ClassicResult.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.LC0_DAG) != 0)
      {
        var lc0DAGResult = Results.LastOrDefault(r => r.EngineName == "LC0_DAG");
        if (lc0DAGResult != null)
          Console.WriteLine($"LC0_DAG       {lc0DAGResult.ChosenMove,10}  {lc0DAGResult.FinalEvaluation,6:F2}  {lc0DAGResult.NumNodes,16:N0}   {lc0DAGResult.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.LC0_DAG_CUDA) != 0)
      {
        var lc0DAGPriorResult = Results.LastOrDefault(r => r.EngineName == "LC0_DAG_PRIOR");
        if (lc0DAGPriorResult != null)
          Console.WriteLine($"LC0_DAG       {lc0DAGPriorResult.ChosenMove,10}  {lc0DAGPriorResult.FinalEvaluation,6:F2}  {lc0DAGPriorResult.NumNodes,16:N0}   {lc0DAGPriorResult.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.CeresMCGS) != 0)
      {
        var resultMCGS = Results.LastOrDefault(r => r.EngineName == "Ceres v2 MCGS");
        if (resultMCGS != null)
          Console.WriteLine($"CeresMCGS     {resultMCGS.ChosenMove,10}  {resultMCGS.FinalEvaluation,6:F2}  {resultMCGS.NumNodes,16:N0}   {resultMCGS.SecondsRuntime,6:F2} sec");
      }
      if ((enginesToUse & TestEngines.CeresMCGS2) != 0)
      {
        var resultMCGS2 = Results.LastOrDefault(r => r.EngineName == "Ceres v2 MCGS2");
        if (resultMCGS2 != null)
          Console.WriteLine($"CeresMCGS2    {resultMCGS2.ChosenMove,10}  {resultMCGS2.FinalEvaluation,6:F2}  {resultMCGS2.NumNodes,16:N0}   {resultMCGS2.SecondsRuntime,6:F2} sec");
      }
      Console.WriteLine();
    }

#if NOT
    /// <summary>
    /// Helper method to create LC0 engine definition.
    /// </summary>
    private static GameEngineDef GameEngineLc0(string testNet, LC0EngineType type, bool disableFutilityPruning = false)
    {
      (string EXE, string extraArgs) lc0Info = type switch
      {
        LC0EngineType.Legacy => (null, ""),
        LC0EngineType.RewriteClassic => SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "")
                                                                : (@"C:\apps\lc031\lc0_plain-preview.exe", ""),
        LC0EngineType.RewriteDAG => SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "dag-preview")
                                                            : (@"C:\apps\lc031\lc0_dag-preview.exe", ""),
        LC0EngineType.TCEC_DAG => SoftwareManager.IsLinux ? ("/home/david/dev/LC0_DAG_TCEC/lc0/build/release/lc0", "")
                                                          : throw new NotImplementedException(),
        _ => throw new NotImplementedException()
      };

      NNEvaluatorDef evalDef1 = NNEvaluatorDef.FromSpecification(testNet, "GPU:0");
      GameEngineDefLC0 engineDefLC1 = new(type.ToString(), evalDef1,
                                          disableFutilityPruning || !MCGSTest.SEARCH_PARAMS_MCGS.FutilityPruningStopSearchEnabled,
                                          null, null,
                                          overrideEXE: lc0Info.EXE,
                                          alwaysFillInHistory: false,
                                          verbose:true,
                                          extraCommandLineArgs: lc0Info.extraArgs,
                                          disableTreeReuse: true);
      return engineDefLC1;
    }

#endif


    /// <summary>
    /// Helper method to create Ceres UCI engine definition.
    /// </summary>
    public static GameEngineDefCeresUCI GameEngineCeresUCI(string testNet)
    {
      static string exeCeres() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net8.0/Ceres.dll"
                                                          : @"C:\dev\ceres\artifacts\release\net8.0\ceres.exe";
      NNEvaluatorDef evalDef1 = NNEvaluatorDef.FromSpecification(testNet, "GPU:0");
      GameEngineDefCeresUCI engineDefCeresUCI = new("CeresUCINew", evalDef1, overrideEXE: exeCeres(),
                                                    disableFutilityStopSearch: !MCGSTest.SEARCH_PARAMS_MCGS.FutilityPruningStopSearchEnabled,
                                                    paramsSearch: MCGSTest.SEARCH_PARAMS_MCTS(MCGSTest.SEARCH_PARAMS_MCGS),
                                                    paramsSelect: new MCTS.Params.ParamsSelect());
      return engineDefCeresUCI;
    }


    /// <summary>
    /// Reset all engines for a new game.
    /// </summary>
    public void ResetEngines()
    {
      ceresEngineNEW?.ResetGame();
      ceresEngineNEW2?.ResetGame();
      ceresEngineClassic?.ResetGame();
      lc0EngineDAG?.ResetGame();
      lc0EngineClassic?.ResetGame();
      ceresEngineProduction?.ResetGame();
    }


    /// <summary>
    /// Dispose of all engines.
    /// </summary>
    public void Dispose()
    {
      ceresEngineNEW?.Dispose();
      ceresEngineNEW2?.Dispose();
      ceresEngineClassic?.Dispose();
      lc0EngineDAG?.Dispose();
      lc0EngineClassic?.Dispose();
      ceresEngineProduction?.Dispose();
    }
  }
}
