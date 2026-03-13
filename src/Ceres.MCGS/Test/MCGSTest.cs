#region Using directives

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AutoMapper.Configuration.Annotations;
using Ceres.APIExamples;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Base.Environment;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.Data.Nets;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.ONNXRuntime;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.NNEvaluators.TensorRT;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;
using Ceres.Commands;
using Ceres.Features.GameEngines;
using Ceres.Features.UCI;
using Ceres.MCGS.Analysis;
using Ceres.MCGS.EngineTests;
using Ceres.MCGS.Environment;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;
using Ceres.MCGS.MCTSNodes.Unused;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Test.RPO;
using Ceres.MCGS.Tests;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.MTCSNodes;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Test;


public partial class MCGSTest
{
  // NOTE: Running SinglePositionMiscTests with validation enabled
  //       often succeeds without any failed assertion.
  //       However about 1/3 of the time we get a rare error:
  //         "node with N=0 and Q not NaN (rather, 0)"
  //       This can possibly occasionally also be reproduced in games of 300 nodes/move.
  //       This is believed to be so rare as mostly inconsequential but should be investigated.
  const bool RUN_VALIDATION = false;

  public static void Main(string[] args)
  {
    // --worker mode: persistent TCP worker for distributed SPSA/NES tuning
    if (Array.IndexOf(args, "--worker") >= 0)
    {
      Ceres.MCGS.Worker.WorkerLocalConfig localConfig = null;
      int? gpuOverride = null;
      int? portOverride = null;
      string hostOverride = null;

      for (int i = 0; i < args.Length; i++)
      {
        if (args[i] == "--worker-config" && i + 1 < args.Length)
          localConfig = Ceres.MCGS.Worker.WorkerLocalConfig.Load(args[i + 1]);
        if (args[i] == "--gpu"  && i + 1 < args.Length) gpuOverride  = int.Parse(args[i + 1]);
        if (args[i] == "--port" && i + 1 < args.Length) portOverride = int.Parse(args[i + 1]);
        if (args[i] == "--host" && i + 1 < args.Length) hostOverride = args[i + 1];
      }

      // CLI overrides take precedence over config file values; fall back to defaults if neither given
      int gpuId    = gpuOverride  ?? localConfig?.GpuId    ?? 0;
      int port     = portOverride ?? localConfig?.Port     ?? 5100;
      string host  = hostOverride ?? localConfig?.BindHost ?? "0.0.0.0";

      Ceres.MCGS.Worker.WorkerServer.LaunchWorkerAsync(gpuId, port, localConfig, host).GetAwaiter().GetResult();
      return;
    }

    // --config for tournament runner
    for (int i = 0; i < args.Length; i++)
    {
      if (args[i] == "--config" && i + 1 < args.Length)
      {
        Ceres.MCGS.GameEngines.SPSATournamentRunnerMCGS.RunTournament(args[i + 1]);
        return;
      }
    }

// for DGX Spark    CeresUserSettingsManager.LoadFromFile(@"/home/david/Ceres.json");
    //    CpuAffinity.LimitProcessToCoreRange(0, 15); probably works, but could have side effects?
    DoMain(args); return;
  }


  // =========================================================
  static string SF17_1_EXE => SoftwareManager.IsLinux ? @"/home/david/apps/SF/sf17.1/stockfish17.1-ubuntu-x86-64-avx2"
                                                      : @"\\synology\dev\chess\engines\stockfish17.1-windows-x86-64-avx2.exe";
  
  static GameEngineDef MakeEngineDefStockfish(string id, string exePath, int numThreads, int hashtableSize = -1)
  {
    return new GameEngineDefUCI(id, new GameEngineUCISpec(id, exePath, numThreads,
                                hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH));//, uciSetOptionCommands: extraUCI);//);
  }
  static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
  static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > (256L * 1024 * 1024 * 1024)
                                                              ? 4096 : 512;
  // =========================================================

  public static void DoMain(string[] args)
  {
    if (System.IO.File.Exists("Ceres.json"))
    {
      CeresUserSettingsManager.LoadFromFile("Ceres.json");
    }

    MCGSLaunch.Launch(args);

    EnvironmentInit();
    ConsoleUtils.WriteLineColored(ConsoleColor.Blue, RuntimeInformation.FrameworkDescription);

    //      ConvertTest();


    //RPOTests.Test(); System.Environment.Exit(3);

    if (false)
    {
      using (new TimingBlock("Bloom"))
        BloomFilterDemo.TestBloomFilterLongVersion();
      System.Environment.Exit(3);
    }   

    if (false)
    {
      CountHashCollisions.Test();
      System.Environment.Exit(3);
    }

    // ********** REPRODUCTION OF BUG WITH INITIAL en passant rights
    //            not being detected in EncodedTrainingPositionReaderTAR.
    //            Will quickly crash.
    if (false)
    {
      // 2r5/5kbp/2P3p1/8/pP2p2P/P3BpP1/5P2/1R5K b - - 0 1 moves a4b3 b1b3 c8c6 b3b7 f7g8 b7b4 g7f8
      IEnumerable<PositionWithHistory> positions = new EncodedTrainingPositionReaderTAR(@"d:\tar\training-run1-test80-20240531-1317.tar")
    .EnumeratePositions()
    .Select(p => p.ToPositionWithHistory());

      foreach (PositionWithHistory loopTestPos in positions)
      {
        foreach (PositionWithMove pp in loopTestPos.PositionsWithMoves)
        {
          MGMove thisMoveMG = MGMoveConverter.MGMoveFromPosAndMove(in pp.Position, pp.Move);
        }
      }
    }

    if (false)
    {
      ComparatorMCGSvsMCTS comparator = new();

      void RunLambdaTest(float lambdaSelect, float lambdaBackup, float lambdaPower)
      {
        comparator.gameEngineCeresMCGS.SelectParams.RPOSelectLambda = lambdaSelect;
        comparator.gameEngineCeresMCGS.SelectParams.RPOBackupLambda = lambdaBackup;
        comparator.gameEngineCeresMCGS.SelectParams.RPOLambdaPower = lambdaPower;

        const int NUM_POSITIONS = 250;
        //CompareMCGSvsMCTS(NUM_POSITIONS, lambdaSelect, lambdaBackup, lambdaPower);
        (int, int) compResult = comparator.RunTest(NUM_POSITIONS);

        Console.WriteLine("Done " + lambdaSelect + "/" + lambdaBackup + " power " + lambdaPower);
        Console.WriteLine();
      }


      CeresUserSettingsManager.Settings.EnableCUDAGraphs = false;
      SILENT = true;
      const float STEP = 0.2f;
      const float LAMBDA_POWER = 0.5f;
      
      List<Task> tasks = [];
//        for (float lambdaSelect = 0.2f; lambdaSelect <= 1.0f; lambdaSelect += STEP)
      for (float lambdaSelect = 1.1f; lambdaSelect <= 1.7f; lambdaSelect += STEP)
        {
//          for (float lambdaBackup = 0.1f; lambdaBackup <= 0.8f; lambdaBackup += STEP*0.5f)
        for (float lambdaBackup = 1.1f; lambdaBackup <= 1.7f; lambdaBackup += STEP)
        {
          Console.WriteLine("launch " + lambdaSelect + " " + lambdaBackup + " " + LAMBDA_POWER);

          RunLambdaTest(lambdaSelect, lambdaBackup, LAMBDA_POWER);
//            tasks.Add(Task.Run(() => RunLambdaTest(lambdaSelect, lambdaBackup)));
//            System.Threading.Thread.Sleep(3000);
        }
      }

      Task.WaitAll([.. tasks]);
      System.Environment.Exit(3);
    }

#if NOT
    var pwh = PositionWithHistory.FromFENAndMovesUCI("4k3/2B1b3/4P1P1/1p1pR3/3P2p1/p1P1K3/1P4r1/8 b - - 0 52", "a3b2 g6g7 g2g3 e3e2 g3g2 e2e3");
    bool hasDup = pwh.ContainsDuplicatePosition(p => MGPositionHashing.HashValue96(p.ToMGPosition, Ceres.Chess.PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98));
    Console.WriteLine(hasDup);
#endif

    if (false)
    {
      //      const string PGN = @"c:\temp\good1.pgn";// @"c:\temp\ceres\match_TOURN_MCGS1_RewriteDAG_638876129830208357.pgn";
      //        const string PGN = @"c:\temp\vsmcts.pgn";
      const string PGN = @"c:\temp\vs_classic.pgn";// vsrewrite.pgn";
      PositionsWithHistory pwhs = PositionsWithHistory.FromEPDOrPGNFile(PGN);
      for (int i = 0; i < pwhs.Count; i++)
      {
        PGNGame game = pwhs.GameAtIndex(i);
if (game.Result != PGNGame.GameResult.Draw) continue;

        int gameLen = game.Moves.Count;
        float lastEval1 = game.MovePlayerEvalCP(gameLen - 2);
        float lastEval2 = game.MovePlayerEvalCP(gameLen - 3);

if (Math.Abs(lastEval2) < 50 && Math.Abs(lastEval1) < 50) continue;

        Console.WriteLine($"\r\nGame #{i + 1}  length={gameLen}  {game.Result}  {game.WhitePlayer} vs {game.BlackPlayer} "
                        + $"last evals {lastEval2} {lastEval1}");
        int plyNum = 0;
        foreach (PositionWithHistory pwm in game.Moves.PositionWithHistories.ToArray())
        {
          int dupCount = pwm.FinalPosition.MiscInfo.RepetitionCount;
          float evalCP = plyNum < gameLen - 2 ? game.MovePlayerEvalCP(plyNum) : float.NaN;
          if (dupCount > 0)
          {
            bool whiteIsEven = pwm.InitialPosition.MiscInfo.SideToMove == SideType.White;
            for (int m = gameLen - 2; m>0; m--)
            {
              if (Math.Abs(game.MovePlayerEvalCP(m)) >= 25 )
              {
                if (m%2==0 == whiteIsEven)
                {
                  Console.WriteLine($" last white " + game.MovePlayerEvalCP(m));
                }
                else
                {
                  Console.WriteLine($" last black " + game.MovePlayerEvalCP(m));
                }
                break;
              }
            }
            Console.WriteLine($" ply {plyNum}: {(int)evalCP,5:F2} cp  {dupCount} position found in game {i} (length {gameLen}) of {gameLen} move {pwm.FinalPosition.MiscInfo.MoveNum} : {pwm.FinalPosition.FEN}");
            //            Console.WriteLine(pwm.FENAndMovesString);
          }
          plyNum++;
        }
      }
      //        int maxRepInMoves = moves.Positions.Max(m => m.MiscInfo.RepetitionCount);
      //        Console.WriteLine($"Game {i} : {game}  { maxRepInMoves}");
      System.Environment.Exit(3);
    }


    if (System.Environment.MachineName == "DEV")
    {
      foreach (var f in typeof(GNodeStruct).GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
      {
        Console.WriteLine($"{f.Name,-30} offset = {Marshal.OffsetOf<GNodeStruct>(f.Name)}");
      }
      Console.WriteLine($"\r\nTotal size GNodeStruct = {Marshal.SizeOf<GNodeStruct>()} bytes");

      Console.WriteLine();
      Console.WriteLine($"\r\nTotal size GEdgeStruct = {Marshal.SizeOf<GEdgeStruct>()} bytes");
      foreach (var f in typeof(GEdgeStruct).GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
      {
        Console.WriteLine($"{f.Name,-30} offset = {Marshal.OffsetOf<GEdgeStruct>(f.Name)}");
      }

      uint mcgsPathVisitHash = ObjUtils.CalcTypeLayoutHash(typeof(MCGSPathVisit));
      Console.WriteLine(mcgsPathVisitHash);
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, ("Sizeof MCGSPathVisit " + Unsafe.SizeOf<MCGSPathVisit>()));
//      GNodeStruct.DumpFieldsAndProperties<MCGSPathVisit>(default(MCGSPathVisit));

      Console.WriteLine($"\r\nTotal size GEdge = {Marshal.SizeOf<GEdge>()} bytes");
      foreach (var f in typeof(GEdge).GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
      {
        Console.WriteLine($"{f.Name,-30} offset = {Marshal.OffsetOf<GEdge>(f.Name)}");
      }

      Console.WriteLine($"\r\nTotal size GNode = {Marshal.SizeOf<GNode>()} bytes");
      foreach (var f in typeof(GNode).GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
      {
        Console.WriteLine($"{f.Name,-30} offset = {Marshal.OffsetOf<GNode>(f.Name)}");
      }

      // System.Environment.Exit(3);
    }

#if NOT
    // This probably has no effect
    AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
    {
      try
      {
        Console.WriteLine("Caught");
        CrashDump.WriteDump("ceres_crash.dmp");
        System.Environment.Exit(3);
      }
      catch
      {
        // Fail silently - we're already crashing
      }
    };
#endif

    //StepThru();

    // lc0_dag-preview --no-nodes-as-playouts  --nncache=39200004 -w d:\nets\weights_run3_753723.pb -t 2  --syzygy-paths="i:\sygyzy\5and6man;i:\sygyzy\7man" --verbose-move-stats --score-type=Q
    // ceres.mcgs network = ~T75 *** GOES NaN on WDL after 25 seconds ***

    // lc0_dag-preview --no-nodes-as-playouts  --nncache=39200004 -w d:\nets\weights_run1_811971.pb.gz -t 2  --syzygy-paths="i:\sygyzy\5and6man;i:\sygyzy\7man" --verbose-move-stats --score-type=Q
    // ceres.mcgs network = ~T81

    //TestPositionMultipleEnginesExample(); System.Environment.Exit(3);
//    SinglePositionMiscTests();
    //CompareMCGSvsMCTS();      System.Environment.Exit(3);
//    TestTerminalPlayout();

    //OrtPinnedMemoryManager<byte>.GPT5Try(); return;
    //OrtPinnedMemoryManager<byte>.ORTTestGraphSimpleBroken();
    //OrtPinnedMemoryManager<byte>.ORTTestGraphSimpleWorksNonCERES(); return;

//    SimpleNetTest(); // ****    

    if (false)
    {
      BackendMultiGPUBenchmark.SingleBatchTest(true);
      //BackendMultiGPUBenchmark.SingleBatchTest(false);
      System.Environment.Exit(3);
    }

    //TestWDLSharpening();
    //TestONNX.RunONNXTest(); System.Environment.Exit(3);  
    //TestSuiteRunner.RunAllTests();
    //MCGSSuiteTest.RunSuiteTest(); System.Environment.Exit(3);

    //RunEngineComparisons();

    // Then do as above except for Stockfish, approximately sample code below
    // Don't reference any particular path for Stockfish, find the one one on the system path
    // Show the stockfish version to the Console (I'm expecting 18)
    // Use 10 threads
#if NOT
    static GameEngineDef MakeEngineDefStockfish(string id, string exePath, int numThreads, int hashtableSize = -1)
    {
      return new GameEngineDefUCI(id, new GameEngineUCISpec(id, exePath, numThreads,
                                  hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH));//, uciSetOptionCommands: extraUCI);//);
    }
    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > (256L * 1024 * 1024 * 1024)
                                                                ? 4096 : 512;
#endif
    //SimpleNetTest();
    //TestAllNets(); System.Environment.Exit(3);
    //RunTournamentTest(); // zz 
    LaunchUCI(System.Environment.GetCommandLineArgs(), null, null);

    string TESTM = "avg_combo_384d_24bn3|cudagraphs=true;bf16=false";
    //string TESTM = "~BT4_FP16_TRT|cudagraphs=true";
    //    string TESTM = "C1-512-55-pre1-I8|cudagraphs=true";
    //string TESTM = "C1-640-34-I8.onnx|cudagraphs=true;bf16=true";
    //TESTM = "avg_combo_768_26_8claude_nc-I8|cudagraphs=true";
    //TESTM = "~T1_DISTILL_256_10_FP16_TRT|cudagraphs=true";
    //TESTM = "~BT4_FP16_TRT|cudagraphs=true";
    if (false)
    {
      SearchLimit limit = SearchLimit.SecondsPerMove(10);
      foreach (string device in new string[] { /*"GPU:0#TensorRT16",*/
                                               "GPU:0#TensorRTNative" })
      {
//        using (new TimingBlock(device))
        {
          FeatureBenchmarkSearch.Benchmark(NNEvaluatorDef.FromSpecification(TESTM, device), limit, false, 33);
        }
      }
      System.Environment.Exit(3);
    }

    if (false)
    {
//      NNEvaluator old = NNEvaluator.FromSpecification(@"e:\cout\nets\C2-384-12-beta1-I8.onnx|V2FRAC=0;V1TEMP=1", "GPU:0#CUDA16");
      NNEvaluator old = NNEvaluator.FromSpecification(TESTM, "GPU:0#TensorRTNative");
      for (int i = 0; i < 2; i++)
      {
        //        long s = GC.GetTotalAllocatedBytes();

        FeatureBenchmarkBackend.BackendBench(null, old, firstBatchSize: 1, maxBatchSize: 768);//, explicitStepSize:19);
//        Console.WriteLine(((double)GC.GetTotalAllocatedBytes() - s) / (1024 * 1024));
      }
      System.Threading.Thread.Sleep(2000);
    }

    if (false)
    {
      //      NNEvaluatorOptionsCeres options = new() { EnableCUDAGraphs = false };

      //      NNEvaluator trt1 = NNEvaluator.FromSpecification("C1-256-10|V2FRAC=0;V1TEMP=1;cudagraphs=true", "GPU:0#TensorRTNative"); // buggy: options with { EnableCUDAGraphs = true });
      //NNEvaluator trt2 = NNEvaluator.FromSpecification("C1-256-10|V2FRAC=0;V1TEMP=1", "GPU:0#TensorRTNative");
      string DEVICE = "GPU:0#TensorRTNative";
      NNEvaluator trt1 = NNEvaluator.FromSpecification(TESTM, DEVICE);
      while (true)
      {
        //        long s = GC.GetTotalAllocatedBytes();

        FeatureBenchmarkSearch.Benchmark(NNEvaluatorDef.FromSpecification(TESTM, DEVICE), 
                                         SearchLimit.SecondsPerMove(5), false, 5);
        System.Environment.Exit(3);

        FeatureBenchmarkBackend.BackendBench(null, trt1, firstBatchSize:1, maxBatchSize:768, explicitStepSize:16);
//        Console.WriteLine(((double)GC.GetTotalAllocatedBytes() - s) / (1024 * 1024));
      }
      Console.WriteLine(trt1.Evaluate(Position.StartPosition));
//      Console.WriteLine(trt2.Evaluate(Position.StartPosition));
      System.Environment.Exit(3);
    }
  }


  private static void TestAllNets()
  {
    NNEvaluatorResult Go(string netID, bool bf16)
    {
      TextWriter originalOut = Console.Out;
      try
      {
        Console.SetOut(TextWriter.Null);
        NNEvaluator e = NNEvaluator.FromSpecification(netID + "|cudagraphs=true" + (bf16 ? ";bf16=true" : ""),
                                                      "GPU:0#TensorRTNative");
        NNEvaluatorResult r = e.Evaluate(Position.StartPosition);
        return r;
      }
      finally
      {
        Console.SetOut(originalOut);
      }
    }

    foreach (string net in new string[]
                                        {
                                          "C1-256-10-I8",
                                          "C2-384-12-beta1-I8",
                                          "avg_combo_384d_21bn4",
                                          "C1-512-15-I8",
                                          "C1-640-34-I8",
                                          "C1-768-26-I8",
                                          "~BT4_FP16_TRT",
                                          "~T3_DISTILL_512_15_FP16_TRT"
                                        })
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "\r\n" + net);
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "  " + Go(net, true).ToString());
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "  " + Go(net, false).ToString());
    }
    System.Environment.Exit(3);
  }

  private static void SimpleNetTest()
  {
//    NNEvaluator evaluator = new NNEvaluatorTensorRT(@"/mnt/devd/nets/t1-256x10-distilled-swa-2432500_fp16.onnx",
//    NNEvaluator evaluator = new NNEvaluatorTensorRT(@"d:\nets\BT4-1024x15x32h-swa-6147500.pb.gz_fp16.onnx",
//                                                       ONNXNetExecutor.NetTypeEnum.LC0, EnginePoolMode.Exact, [1, 8, 32, 48, 64, 96, 128, 256]);
//    NNEvaluator evaluator = NNEvaluator.FromSpecification("~BT4_FP16_TRT", "GPU:0,0#TensorRTNative");

    // NNEvaluator evaluator = NNEvaluator.FromSpecification("~BT4_FP16_TRT", "GPU:0#TensorRT");
    //NNEvaluator evaluator = NNEvaluator.FromSpecification("C2-384-12-beta1-i8", "GPU:0#TensorRTNative");
    NNEvaluator evaluator = NNEvaluator.FromSpecification("~T3_DISTILL_512_15_FP16_TRT", "GPU:0#TensorRTNative");

    //NNEvaluator evaluator = NNEvaluator.FromSpecification("~BT4_FP16_TRT", "GPU:0#TensorRT");
    EncodedPositionBatchFlat batch = BuildBatch(1);
    NNEvaluatorResult[] rx = evaluator.EvaluateBatch(batch);
    long ba = GC.GetAllocatedBytesForCurrentThread();
    for (int j=0;j<2;j++)
    {
      using (new TimingBlock("eval"))
      {
        for (int i = 0; i < 1; i++)
        {
          NNEvaluatorResult[] r = evaluator.EvaluateBatch(batch);
          //          IPositionEvaluationBatch xx = evaluator.EvaluateIntoBuffers(batch);
          //      NNEvaluatorResult[] r = evaluator.EvaluateBatch(BuildBatch(1, i % 2 == 1));
          //      Console.WriteLine(r[0].V + " " + r[0].Policy);

          //      NNEvaluatorResult[] r = evaluator.EvaluateBatch(BuildBatch(2, i % 2 == 1));
          Console.WriteLine(r[0].V);// + " " + r[0].Policy);
          Console.WriteLine(r[1].V);// + " " + r[1].Policy);
        }
      }
    }
//    Console.WriteLine((GC.GetAllocatedBytesForCurrentThread() - ba) / (1024 * 1024));
    System.Environment.Exit(2);
  }


  private static void TestWDLSharpening()
  {
    //    NNEvaluator evalCeres = NNEvaluator.FromSpecification("~BT4_FP16_TRT", "GPU:0#CUDA16");
    NNEvaluatorDef netDef = default;

    //    string NET_CERES = "~BT4_FP16_TRT";// "~T3_DISTILL_512_15_FP16_TRT";
    //" --history-fill-new=no";

    string EXTRA_UCI = " --history-fill-new=always"; // "--wdl-calibration-elo=3600 --wdl-eval-objectivity=0 --wdl-draw-rate-reference=0.64";
    string EXTRA_CERES = "|V1TEMP=0.3";

    string T1_CERES = "~T1_256_RL_TRT"; // SimpleLC0Net("t1-256x10-rl-base-swa-3860000.pb.gz")
    string T1_LC0   = "~T1_256_RL_NATIVE"; //ONNXNet16LC0("t1-256x10-rl-base-swa-3860000_fp16", true)

    string NET_CERES = T1_CERES + EXTRA_CERES;
    string NET_Lc0 = T1_LC0;

//    string NET_CERES = "~T3_DISTILL_512_15_FP16_TRT" + EXTRA_CERES;
//    string NET_Lc0 = "t3-512x15x16h-distill-swa-2767500.pb.gz";

    NNEvaluator evalCeres = NNEvaluator.FromSpecification(NET_CERES, "GPU:0#CUDA16");
//evalCeres.ZeroHistoryPlanes = false;

    // C:\apps\lc0_32>lc0_33pre-trt.exe -w d:\nets\weights_run1_811971.pb.gz -t 2 --backend=onnx-trt
    var engineLc0 = GameEngineLc0(NET_Lc0, "GPU:0#CUDA16", LC0EngineType.RewriteDAG, true,
                                  extraUCIOptions:EXTRA_UCI).CreateEngine();

    IEnumerable<PositionWithHistory> positions = new EncodedTrainingPositionReaderTAR(@"d:\tar\training-run1-test60-20210515-1417.tar")
  .EnumeratePositions()
  .Select(p => p.ToPositionWithHistory());

    List<double> ceresQ = [];
    List<double > lc0Q = [];
    ValueWDLOptimalEntropyCalculator entropy = new();

    int count = 0;
    foreach (PositionWithHistory loopTestPos in positions)
    {
      if (ceresQ.Count > 10)
      {
        break;
      }
      if (count++ % 39 != 0)
      {
        continue;
      }
      try
      {
        Position testPos = loopTestPos.FinalPosition;
        NNEvaluatorResult ceresEval = evalCeres.Evaluate(testPos);

#if NOT
        var lc0Eval = engineLc0.Search(loopTestPos, SearchLimit.NodesPerMove(1));
        entropy.AddWDL(ceresEval.W, ceresEval.D, ceresEval.L, 
         lc0Eval.wdl );
#endif
        float lc0V = engineLc0.Search(new PositionWithHistory( testPos), SearchLimit.NodesPerMove(1)).ScoreQ;
        if (Math.Abs(ceresEval.V) < 0.95)
        {
          ceresQ.Add(ceresEval.V);
          lc0Q.Add(lc0V);
          Console.WriteLine(ceresEval.V + " " + lc0V + " " + loopTestPos.FinalPosition.FEN);
        }

      }
      catch (Exception)
      {

      }
    }

    Console.WriteLine(StatUtils.Correlation(ceresQ.ToArray(), lc0Q.ToArray()));
    Console.WriteLine(ValueWDLOptimalEntropyCalculator.MeanAverageAbsoluteDeviation(ceresQ.ToArray(), lc0Q.ToArray()));
    System.Environment.Exit(3);
  }


  private static EncodedPositionBatchFlat BuildBatch(int batchSize, bool flip = false)
  {

    EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;
    EncodedPositionBatchBuilder batchBuilder = new(batchSize, NNEvaluator.InputTypes.All);

    for (int i = 0; i < batchSize; i++)
    {
      Position testPos = i % 2 == (flip ? 1 : 0) ? Position.StartPosition 
                                                 : Position.FromFEN("4k3/1P3p2/8/7p/rB2pb2/2P4P/2K5/6R1 b - - 0 50");
testPos = Position.StartPosition;
      batchBuilder.Add(testPos);
    }
    EncodedPositionBatchFlat batch = batchBuilder.GetBatch();
    return batch;
  }





  /// <summary>
  /// Example usage of TestPositionMultipleEngines class.
  /// </summary>
  public static void TestPositionMultipleEnginesExample()
  {
    List<(string fen, SearchLimit limit, string description, string correctMove, EPDEntry epd)> testPositions = [];

    string TEST_EPD = @"C:\Users\ellio\Downloads\failedLichessPuzzles_2025-07-15_01-39X.epd";
    TEST_EPD = null;// @"z:\chess\data\epd\hard-talkchess-2022.epd";

    const int MAX_EPD = 100;
    SearchLimit limitEPD = SearchLimit.SecondsPerMove(30f);

    if (TEST_EPD != null)
    {
      //      foreach (PositionWithHistory pwh in PositionsWithHistory.FromEPDOrPGNFile(TEST_EPD))
      foreach (EPDEntry epd in EPDEntry.EPDEntriesInEPDFile(TEST_EPD))
      {
        if (testPositions.Count >= MAX_EPD)
        {
          break;
        }
        //        string moveSAN = MGMoveConverter.ToMove(moveMG).ToSAN(SearchRoot.CalcPosition().ToPosition);
        //        var correctness = epd.CorrectnessScore(, 10)
        testPositions.Add((epd.PosWithHistory.FENAndMovesString, limitEPD, "test", epd.BMMoves[0], epd));
      }
    }

#if NOT
match SF vs Ceres v2 (640-34) @240+4/120+2 

1.endgame (at two points), can't see drawn (either of these two places)
1k1rr3/pppn1ppp/4p3/2P3P1/1P1P4/P2R4/4N1PP/5RK1 b - - 0 28 
8/ppn1k2p/2p3p1/2P1PpP1/1P3N1P/P2K2P1/8/8 w - - 4 40 


3. can't see lost in QRN endgame
7k/6pp/pq6/5Q2/R7/2N2PK1/1r1rN1PP/8 b - - 4 36 

9. can't see drawn in 2N2B endgame
8/pp4k1/4nn2/8/P6K/3B4/5P2/B7 w - - 0 40 

26. can't see massively lost
1r1k3r/q2p1Qpp/1p1p1p2/5P2/8/7P/P4RP1/2R3K1 b - - 1 31 

?. thinks somewhat winning, actual dead draw BNP endgame
8/5kp1/2p1p3/5n1p/1P1P1K2/2B5/1P3PP1/8 b - - 0 40 

-------------
now using T81 not C1-640-34-i8
?. can't see lost, trending toward draw (Q=-.328)
6k1/1p1q4/3p4/p2P1Nn1/P1r4p/3Q2P1/5RK1/8 b - - 1 41 
info depth 43 seldepth 101 time 363000 nodes 20600308 score cp -41 tbhits 99540 nps 56750 pv c4b4 d3e3 h4h3 g2h2 g5e4 f2f3 b4b2 h2h3 b2b1 f3f4 d7h7 f5h4 b1b4 e3c1 e4c5 f4f6 h7h5 f6g6 g8h7 g6g5 h5h6 h3g2 b4b2 g2g1 h6f6 g1h1 b2b4 h4f5 b7b6 g5g7 h7h8 c1f1 f6g7 f5g7 h8g7 f1f5 b4a4 g3g4 a4b4 f5g5 g7f7 g5h5 f7g7 g4g5 c5e4 h5h6 g7g8 g5g6 e4f6 g6g7 f6h7 h1h2 b4b2 h2g3 b2b3 g3f4 b3b1 h6d6 b1f1 f4e3 f1f7 d6b8 g8g7 b8g3 g7h8 g3g6 f7a7 g6b6 a7d7 b6b8 h8g7 b8g3 g7f7 e3d4 a5a4 d5d6 h7f6 g3f4 d7a7 f4d2 a4a3 d2a2 f7g6 d4c5 g6g7 c5b5  string M= 91
lc0-dag has same problem! info depth 50 seldepth 108 time 318602 nodes 12588113 score cp -64 wdl 105 507 388 nps 39958 tbhits 104564 pv c4b4 d3e3 h4h3 g2h1 g5e4 

#endif
    // ...............................................................................................................
    // approximately equal: C1-512-15 vs ~BT2_NATIVE (for Lc0)
    //    string NET_CERES = "~BT4_FP16_TRT";//|cudagraphs=true";// "~T3_DISTILL_512_15_FP16_TRT";
    string NET_CERES = "~BT4_FP16_TRT|cudagraphs=true";// "~BT5_FP16_TRT"; // ~BT4__PT332_FP16_TRT
                                        //    string NET_CERES = "~T3_DISTILL_512_15_FP16_TRT";
                                        //    string NET_LC0 = "~T3_DISTILL_512_15_NATIVE";
                                        //    string NET_CERES = "~T1_DISTILL_256_10_FP16_TRT";
                                        //    string NET_LC0 = "~T1_DISTILL_256_10_NATIVE";

    string NET_LC0 = "~BT4_NATIVE";// NET_CERES.Contains("BT4") ? "~BT4_NATIVE" : NET_CERES;
//    NET_LC0 = NET_LC0.Contains("BT4_PT332") ? "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz" : NET_LC0;
//    NET_LC0 = NET_LC0.Contains("~T3_DISTILL_512_15_FP16_TRT") ? "t3-512x15x16h-distill-swa-2767500.pb.gz" : NET_LC0;
//    NET_LC0 = NET_LC0.Contains("~T3_DISTILL_512_15_FP16_TRT") ? "t3-512x15x16h-distill-swa-2767500.pb.gz" : NET_LC0;

    const bool VERBOSE = true;
    //NET_CERES = "C1-640-34-i8|cudagraphs=true";
    //TestEngines TEST_ENGINES = TestEngines.LC0_DAG | TestEngines.LC0_DAG_CUDA;
    //TestEngines TEST_ENGINES = TestEngines.All;
    TestEngines TEST_ENGINES = TestEngines.CeresMCGS | TestEngines.LC0_DAG;//| TestEngines.CeresMCGS2;
    SearchLimit LIMIT = SearchLimit.SecondsPerMove(90);// 7200/(21*3));
    LIMIT = SearchLimit.NodesPerMove(3_000_000);
    // ...............................................................................................................

    //          ("n1b1rkr1/ppq1pp1p/4n3/3p1R2/1P6/4P3/PQ1P2PP/1B1NN1KR w He - 3 10", LIMIT, "rook sacrifice, Chess960 castling needed", null, null),

    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Using neural network: " + NET_CERES);  
    using (TestPositionMultipleEngines tester = new(NET_CERES, NET_LC0))
    {
      if (TEST_EPD == null)
      {
        const string FEN_BISHOP_ENDGAME_LOST = "6k1/5p2/p5p1/P1B2p2/1P5p/3b3P/7K/8 w - - 0 44";
        const string FEN_KOVAX_DRAW_NAVS = "R7/P7/7p/7p/5P1k/8/r5K1/8 w - - 0 1"; // See Lc0 forum post "Kovax recently suggested an interesting endgame position..."
                                                                                  // Test a few different positions
        //SearchLimit LIMIT = SearchLimit.NodesPerMove(500_000);
        
        testPositions =
        [
          ("r4k2/2p2p2/bp3q1b/p2Pp3/2P1Pp2/PPN4Q/5P2/3B1KR1 b - - 3 33", LIMIT, "CeresvSF doesn't see lost", null, null),
          //("8/rB3R2/8/8/8/bK6/3B4/6k1 w - - 0 29", LIMIT, "7-man no pawn, 1.6mnps", null, null),
//          ("8/3Q4/3p1bk1/3Pp3/4Pn1p/8/8/3K4 w - - 0 5", LIMIT, "Dead lost, BT5 recognizes less than Ceres (300cp vs 645cp)", null, null)
//          ("8/1p3k2/2p3p1/2P1Q2p/1P5P/5KP1/5P2/2q5 b - - 2 61", LIMIT, "Ceres saves as draw vs Rubi, swindle?", null, null),
//          ("r3kb1r/p4ppp/2p1bn2/qp1p4/P2QP3/2N1BP2/1PP2P1P/2KR1BR1 b kq a3 0 13", LIMIT, "Ceres loss to Renegade, b4 bad", null, null),
//          ("r1b1kb1r/pp3ppp/2p2n2/q2p4/3QP3/2N1BP2/PPP2P1P/2KR1B1R b kq - 1 11", LIMIT, "Ceres loss to Renegade, b4 bad", null, null),
//
//          ("1n1q3k/2p1b2p/p3p1p1/2PpPp2/P2P1P2/5NQP/nr1B1PK1/3B2R1 w - - 3 93", LIMIT, "Middlegame, hard to see winning", null, null),
#if FIRST

         ("8/1q5k/6p1/K2p2Pp/1P1Q3P/8/8/8 b - - 1 57", LIMIT, "Lepned queen endgame, lost, run up to 50mm nodes!", null, null),
          ("7r/p1pk2bp/1pnpq1p1/3N4/5B1P/1Q2P3/PPP3P1/2KR4 w - - 0 23", LIMIT, "TCEC Swiss 9 first game, can't see lost", null, null),
          ("r6k/1prqn1p1/2n1p2p/3pPp1P/pPbP3N/P1B3R1/3Q1PP1/KBR5 b - - 8 27", LIMIT, "Ceres T79 slow to see lost", null, null),
          (Position.StartPosition.FEN, LIMIT, "startpos", null, null),
          ("2rq1rk1/1b1nbppp/p2pp3/np4P1/3NPP1P/2N1B3/PPPQ2B1/1K1R3R w - - 5 15", LIMIT, "Kovax opening, hard, very deep search", null, null),
          ("3r3r/1b3p2/3k3p/1p1ppP1K/pP4PP/2P2B2/3R4/3R4 b - - 3 50", LIMIT, "Mistakenly thinks draw (hard)", null,null),
          ("4r1r1/1b3p2/3k1p1p/1p1p1P1K/pP2P1PP/2P2B2/8/3RR3 b - - 6 47", LIMIT, "Ceres bug_fails_see_win,Lc0 slow!", null, null),
          ("6k1/1b3p1p/2q1p1p1/2P5/4PP2/4Q2P/r2N2PK/2R5 b - - 0 30", LIMIT, "endgame Ceres 4xgpu slow to see lost", null, null),
          ("6r1/2p1k1r1/pp1p4/3Pp2p/2P2p1P/P2P1P2/2P2KP1/4R1R1 w - - 1 67", LIMIT, "Rook ending, SF slow to see winning", null, null),
          ("1Q6/5p2/p3rkp1/P1P1np2/7P/6K1/6P1/8 w - - 23 69", LIMIT, "best game TCEC, see won?", null, null),
          ("2r1k1r1/3qbp1p/1p2p1pB/p1npP2P/P2N2Q1/1PP5/5PP1/R3R1K1 b - - 0 23", LIMIT, "difficult middlegame", null, null),

         ("1k6/8/2Q5/1P6/5p1p/4q2P/6P1/5K2 w - - 0 80", LIMIT, "queen endgame, lost", null, null),
          ("4r3/p3Brpk/1n2R3/3p4/3P4/5N2/P5PP/R5K1 w - - 0 23", LIMIT, "CeresV2 crash", null, null),
          ("3R4/8/r7/6pk/7p/6bP/6P1/3R2K1 b - - 0 42", LIMIT, "Endgame CeresV2 thinks won, not Lc0", null, null),
          ("8/3bk3/3p1pB1/2pPpP2/1nP1P3/1P2K3/8/5N2 w - - 0 42", LIMIT, "Ceres vs SF, fortress endgame, can't see draw", null, null),
          ("4nr2/5ppk/pn3q2/3ppNRP/1pr1P3/1P3P2/1KP3Q1/R7 w - - 0 42", LIMIT, "Kovax slow endgame, hard, g5g6 wins", null, null),
          ("4K3/2k1Bp1N/6p1/5PP1/8/7p/b7/8 w - - 0 0", LIMIT, "Endgame puzzle, e7f6 only draw, fortress", null, null),
          ("8/6k1/2PpPnp1/P6p/3K1P1P/6P1/8/8 b - - 0 50", LIMIT, "early endgame", null, null),
          (FEN_BISHOP_ENDGAME_SHOULD_BE_DRAW, LIMIT, "Bishop endgame should be draw", null, null),
          ("8/6p1/7p/2k4P/2P1N1P1/3K4/1b6/8 b - - 10 61", LIMIT, "late B vs N endgame, draw", null, null),
          ("b2r3r/4Rp1p/p2q1np1/kp1P4/3Q4/P4PPB/1PP4P/1K6 w - - 0 4 moves b4 a5a4 e7a7 a8b7 d4c3 d6d5 a7b7 d5c4 c3f6 d8d1 b1b2 h8a8 f6b6", LIMIT, "b4!! Lc0 immediate sees", null, null),
          ("2rq1rk1/1b1nbppp/p2pp3/np4P1/3NPP1P/2N1B3/PPPQ2B1/1K1R3R w - - 5 15", LIMIT, "Kovax opening, hard, Qf2!", null, null),
          ("6r1/p1B2p1k/2BP3p/1Pb2bp1/2q3P1/3p3P/3Q1P2/5RK1 b - g3 0 37", LIMIT, "Middegame, dead lost, C640-34 slow to see", null, null),

         ("2k5/1p3p2/nP1p2p1/3N1p2/2PP1P2/6Pp/4K2P/8 b - - 0 42", LIMIT, "K+P endgame, white +3, takes Stockfish a few seconds (ask LLM)", null, null),
          ("8/8/6p1/3k4/P5pP/1PK1B1P1/5P2/r7 b - - 0 54", LIMIT, "Endgame, dead lost", null, null),
          ("1r3r1k/p2nbpp1/2n1p2p/q1ppP2P/3P4/2PBB2Q/P3NPP1/R2R2K1 w - - 0 23", LIMIT, "middlegame,difficult,only +93cp", null, null),
          ("2k5/8/3p4/3Pp1p1/4P1Pn/2K2P2/8/6N1 w - - 10 4", LIMIT, "fortress, N endgame, draw", null, null),
          (FEN_KOVAX_DRAW_NAVS, LIMIT, "Kovax unrecognized draw (easy)", null, null),
          (FEN_TCEC_EMBARRASING_ENDGAME_LOSS, LIMIT, "TCEC EMB post", null, null),
          ("4k3/7p/1pn1p1r1/p7/P7/1P4PR/1B3P2/6K1 b - - 0 45", LIMIT, "very difficult endgame", null, null),
          ("8/p7/r7/3k1N2/2p2K2/8/P6R/8 b - - 2 59", LIMIT, "don't play 16f6", null, null),


          ("3k4/5Rpp/6r1/1p1p4/4pP1P/1N2K1P1/1P6/8 b - - 0 1", LIMIT, "Should be draw, MCGS worse", null, null),
          (FEN_SEE_LOST_TRY_2MM, LIMIT, "See lost 2mm", null, null),
#endif
#if PART1
         ("8/3k2p1/8/5K1P/4P3/2b5/2P1B3/8 w - - 15 66", LIMIT, "Lepned crash, SF+1, maybe not draw?", null,null),
          //  *** Ceres MCGS fails badly on this one; fails to converge toward draw fast; ***
          // it seems this is probably due to "50 move rule" being necessary to recognize

//          (FEN_TCEC_EMBARRASING_ENDGAME_LOSS3, LIMIT, "TCEC EMB post3", null, null),
          (FEN_BISHOP_ENDGAME_LOST, LIMIT, "Bishop endgame lost", null, null),

//          ("8/1p3ppk/4p3/4P2P/2bR1B2/3p1P1K/2r3P1/8 w - - 10 43", LIMIT, "Easy draw", null, null),
//          (FEN_TCEC_EMBARRASING_ENDGAME_LOSS2, LIMIT, "TCEC EMB post2", null, null),

//          ("r4k2/8/5p2/7p/3PPb1P/4N3/5KR1/8 b - - 1 36", SearchLimit.SecondsPerMove(30), "TCEC embarrasing", null, null),
//          ("8/6k1/7p/4P3/3K4/6r1/2p2r2/2R4R w - - 9 63", SearchLimit.SecondsPerMove(7), "Rh1h4 loses", null, null),

//            ("r4k2/2p2p2/1p1p2p1/P2Pn1Pp/R6P/K1P5/8/1R6 b - - 2 38", SearchLimit.SecondsPerMove(10), "Lepned bug with redescent2", null, null),
///          ("8/8/4kp2/P1Bp2p1/3P4/5P1K/4b2P/8 b - - 73 79", SearchLimit.SecondsPerMove(5), "CeresV2 loss at 20+2, not Ke6f5??", null, null),
//            ("5b2/1pQ1kpp1/6p1/p1pq4/8/6BP/1P3PPK/8 b - - 0 1 moves e7f6 c7b8 f8e7 b8f4 f6e6 f4g4 d5f5", SearchLimit.NodesPerMove(15000), null, null,null)
///            ("8/2q1k3/5N2/1p1P3N/p7/2P5/PBK5/8 w - - 0 1", SearchLimit.NodesPerMove(500_000), "Lepned long term queen trap that ends in a tablebase win when the queen has to be sacrificed for a knight, see around 500k", null, null),
//            ("8/7p/3kp1p1/p1p5/4KP2/1P6/1P4PP/8 w - - 0 30", limit, "", null, null),
//            ("8/2bBnpk1/2p5/2P1p1pq/4P3/4B1PP/5K2/Q7 w - - 7 42", SearchLimit.NodesPerMove(1_000_000), "", null, null)
        //          ("4B3/8/7P/KR6/2p5/8/2k5/1r6 w - - 0 1", limit, "", null),
        //          (FEN_MATE_IN_2, limit, "", null),
        //          (FEN_NICE_EXAMPLE, limit, "", null),
//         ("8/1p6/1p3k2/3p2P1/1P1PpKP1/8/P7/8 b - - 0 39", LIMIT, "Zugzwang lepned", "f6g7", null) // maybe not bug, eventually converges
#endif

        ];
      }

      foreach ((string fen, SearchLimit limit, string description, string correctMove, EPDEntry epd) testPos in testPositions) 
      {
        PositionWithHistory position = PositionWithHistory.FromFENAndMovesUCI(testPos.fen);

        try
        {
          tester.TestPosition(position, testPos.limit,
                              TEST_ENGINES,
                              testPos.description, testPos.correctMove, testPos.epd, 
                              VERBOSE, RUN_VALIDATION);
        }
        catch (Exception exc)
        {
          Console.WriteLine(exc);
        }

        // Reset engines between tests
        tester.ResetEngines();
      }

      // Show accumulated results
      Console.WriteLine("\n=== ACCUMULATED RESULTS ===");
      string last = "";
      int wrongCeresProd = 0;
      int wrongCeresMCGS = 0;
      int wrongCeresMCGS2 = 0;
      int wrongLc0_DAG = 0;
      int wrongCeresMCTS = 0;
      int wrongeLc0Classic = 0;
      int count = 0;
      foreach (TestPositionResult result in tester.Results)
      {
        if (result.Position.FENAndMovesString != last)
        {
          count++;
          Console.WriteLine();
        }
        string wrongStr = (result.CorrectMove != null && result.CorrectMove != result.ChosenMove) ? "X" : " ";

        ConsoleColor color = ConsoleColor.White;
        if (result.EngineName == "Ceres v2 MCGS" && wrongStr == "X")
        {
          color = ConsoleColor.Red;
          wrongCeresMCGS++;
        }
        if (result.EngineName == "Ceres v2 MCGS2" && wrongStr == "X")
        {
          color = ConsoleColor.Red;
          wrongCeresMCGS2++;
        }
        else if (result.EngineName == "CeresProd" && wrongStr == "X")
        {
          wrongCeresProd++;
        }
        else if (result.EngineName == "LC0_DAG" && wrongStr == "X")
        {
          color = ConsoleColor.Yellow;
          wrongLc0_DAG++;
        }
        else if (result.EngineName == "Ceres v2 MCTS" && wrongStr == "X")
        {
          wrongCeresMCTS++;
        }
        else if (result.EngineName == "LC0_Classic" && wrongStr == "X")
        {
          wrongeLc0Classic++;
        }

        string info = result.UCIInfo ?? "";
        string pv = "";
        int indexPV = info.IndexOf("pv ", StringComparison.OrdinalIgnoreCase);
        if (indexPV > 0)
        {
          pv = info[(indexPV + 3)..];
        }
        int nps = (int)Math.Round(result.NumNodes / result.SecondsRuntime);
        string infoTruncated = info.Replace("info ", "")
                                   .Replace("depth", "d")
                                   .Replace("seldepth", "sd")
                                   .Replace("score ", "")
                                   [..Math.Min(info.Length, 82)];
        ConsoleUtils.WriteLineColored(color, $"{result.EngineName,-13} {result.CorrectMove,-6}  {wrongStr} {result.ChosenMove,-7} " 
                                  + $"{result.FinalEvaluation,6:F2}  {result.NumNodes,10:N0}  {result.SecondsRuntime,6:F2}s  {nps,8:N0} nps " 
                                  + $"{result.CUDAUtilization.GetStats().AvgGpuUtilPct,6:F2}%   {result.Description,-20} " 
                                  + $"{infoTruncated}  {pv}  {result.Position.FENAndMovesString}");
        last = result.Position.FENAndMovesString; 
      }

      Console.WriteLine();
      Console.WriteLine($"Total positions   : {count}");  
      Console.WriteLine($"Wrong CeresProd   : {wrongCeresProd}");
      Console.WriteLine($"Wrong CeresMCTS   : {wrongCeresMCTS}");
      Console.WriteLine($"Wrong CeresMCGS   : {wrongCeresMCGS}");
      Console.WriteLine($"Wrong CeresMCGS2  : {wrongCeresMCGS2}");

      Console.WriteLine($"Wrong LC0_Classic : {wrongeLc0Classic}");
      Console.WriteLine($"Wrong LC0_DAG     : {wrongLc0_DAG}");

      // Output one line for each engine, show the name and total seconds and total nodes to console
      Console.WriteLine();
      var engineTotals = tester.Results
                               .OrderBy(r=>r.EngineName)
                               .GroupBy(r => r.EngineName)
                               .Select(g => new
                               {
                                 EngineName = g.Key,
                                 TotalSeconds = g.Sum(r => r.SecondsRuntime),
                                 TotalNodes = g.Sum(r => (long)r.NumNodes),
                                 GPUPct = g.Sum(r => r.CUDAUtilization.GetStats().AvgGpuUtilPct) / g.Count()
                               });
                               //.Sort(r => r.EngineName);


      foreach (var totals in engineTotals)
      {
        Console.WriteLine($"{totals.EngineName,-15} Time: {totals.TotalSeconds,8:F2}s  Nodes: {totals.TotalNodes,15:N0}  GPU {totals.GPUPct,6:F2}%");
      }
    }
  }
  

  // Mate in 2 involving queen sacrifice, takes 100 to 200 nodes to discover e5h8.
  const string FEN_MATE_IN_2 = "r5k1/pppn1r1p/1b5q/4Q1N1/3P1P2/2P5/PP5P/2K3R1 w - - 1 25";

  // TODO: This is a buglet, if using MCGS engine then starting from a drawn by repetition does not return any legal moves.
  const string FEN_BUG_DRAW_REPETITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const string FEN_BUG_DRAW_REPETITION_MOVE = "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 b7b6 h2h4 d8d7 h1h3 c8a6 a2a3 b4c3 h3c3 a6f1 e1f1 g8e7 g1e2 b8c6 b2b4 e8c8 a3a4 c6b4 a4a5 c8b7 c1a3 b4c6 a3e7 c6e7 a5b6 a7b6 c3a3 d8a8 d1d3 h7h5 a3a6 e7c6 a6a8 h8a8 a1a8 b7a8 g2g3 a8b7 e2f4 g7g6 c2c3 c6e7 f4h3 e7f5 h3g5 f5h6 f1g2 d7e8 d3c2 e8b5 g5f7 h6f7 c2g6 f7d8 g6h5 b5d3 h5e8 d3e4 g2h2 b7c8 h4h5 e4f3 h5h6 f3f2 h2h3 f2f7 e8h8 c8d7 h6h7 f7h5 h3g2 h5e2 g2g1 e2d1 g1g2 d1c2 g2g1 c2c1 g1g2 c1d2 g2g1 d2e3 g1g2 e3e4 g2g1 d8f7 h8g7 e4e1 g1g2 e1e2 g2g1 e2d1 g1g2 d1e2";

  const string FEN_FEN_BLACK_FORCED_MOVE_WHITE_MULTIPLE_MATES = "7k/4B1p1/6B1/7Q/q7/8/r5r1/4K3 b - - 0 1";


  // Blockade position (graph with only about 10 * 30 = 300 nodes).
  const string FEN_BLOCKADE_POSITION = "3k4/8/8/8/1p1p1p1p/pPpPpPpP/P1P1P1P1/6K1 w - - 0 1";
  const string FEN_LockedPosition384Reachable = "4k3/8/8/1p1p1p1p/pPpPpPpP/P1P1P1P1/8/4K3 w - - 0 1";
  const string FEN_LockedPosition256Reachable = "4k3/p1p1p1p1/PpPpPpPp/1P1P1P1P/8/8/8/4K3 w - - 0 1";
  const string FEN_LockedPosition16Reachable = "2k1b3/1p1p1p2/1P1P1P2/8/8/2p1p1p1/2P1P1P1/3B1K2 w - - 0 1";


  const string FEN_SHOULD_BE_DRAW_QUICKLY_SEE = "8/5Rp1/7p/5K1P/2kP2P1/2p5/2P1r3/8 w - - 7 1";

  // best:
  // 8/8/4npk1/7p/R4P1P/5K2/8/8 b - - 0 50 // 
  // 2R5/p3r3/P4p1k/1P4p1/6P1/8/5K2/8 w - - 5 65 (?) should steadily increase toward +1.00

  // Endgame position lost by Ceres, plays inferior f4d3 instead of f8f7
  // Takes about 1.1mm positions for Lc0/BT4 DAG (not found in 5mm by non-DAG)
  const string FEN_TCEC_EMBARRASING_ENDGAME_LOSS = "r4k2/8/5p2/7p/3PPb1P/4N3/5KR1/8 b - - 1 36";
  const string FEN_TCEC_EMBARRASING_ENDGAME_LOSS2 = "8/5k2/5p2/5N1p/3PPb1P/8/6K1/8 b - - 0 39";   // a few moves down the PV, should be draw!
  const string FEN_TCEC_EMBARRASING_ENDGAME_LOSS3 = "8/5k2/5p2/3P1N1p/4P2P/2b5/5K2/8 b - - 0 41";

  const string FEN_NEG89CP_BG5_ONLY = "8/4Bpk1/p5p1/P4p2/1P5p/3b3P/8/6K1 w - - 2 45";

  const string FEN_BISHOP_ENDGAME_SHOULD_BE_DRAW = "8/4k3/8/4pK2/3bPP2/1B4P1/8/8 b - - 0 7"; // hard, SF oscillating briefly still at 5sec

  const string T75_600K_D2D5_LOSES = "8/p7/2n5/2p1bNPk/p3P3/1r5p/1P1RK2P/2B5 w - - 4 41";

  // Position from Ceres vs Beserk
  // Stockfish realizes a draw after a few seconds (so it's actually not that easy)
  // Lc0 DAG drives down to 0.26 in a few million, Ceres DAG still at 0.41 at 25mm
  const string FEN_UNRECOGNIZED_ENDGAME_DRAW = "5k2/6n1/8/6Pp/4KP2/2b5/8/3R4 w - - 0 1";

  const string ROOK_VS_KNIGHT_CANT_SEE_BLACK_LOST = "8/8/4npk1/7p/R4P1P/5K2/8/8 b - - 0 50";
  const string FEN_DANIEL_ONE_MOVE_DRAWS = "2R5/p4pk1/6p1/3p3p/3Pp2P/p3r1PK/5r2/8 w - - 0 40";
  const string FEN_ENDGAME_ROOK_WHITE_WINNING_LC0_SLOWER_NO7MAN = "8/5p2/8/4K2k/1PR5/7P/8/1r6 b - - 5 51";

  const string FEN_H2H3_CERES_MCGS_SLOWER_THAN_LC0_DAG = "4R3/8/p7/1p3k2/2pp1r1p/P7/1P4KP/3B4 w - - 0 54";
  const string FEN_CERES_MCGS_SEES_WINNING_FASTER_THAN_LC0 = "7k/1R6/7P/4n3/1Pb5/6K1/8/8 w - - 8 59";
  const string FEN_SEE_LOST_TRY_2MM = "1rq4k/4p2p/3p1p2/3P1NpN/pr2nPQ1/8/1P1nR1PK/4R3 w - - 4 41";

  //  Ceres sees there are 4 winning moves much more quickly than Lc0 DAG
  const string FEN_LC0_DAG_CANT_SEE_WIN = "2R5/p3r3/P4p1k/1P4p1/6P1/8/5K2/8 w - - 5 65";

  // Nice position showing value of DAG (e.g. T3D)
  // Unintuitive d6+ wins! (b2a3 bad), takes a while to realize
  //   Ceres/C1-640-34  1000k nodes
  //   Lc0/BT4          1100k nodes
  //   Lc0-DAG/BT4      310k nodes
  const string FEN_NICE_EXAMPLE = "8/2q1k3/5N2/1p1P3N/p7/2P5/PBK5/8 w - - 0 1";

  const string RA4_WINNING = "3br1k1/p4rp1/1p2p1Q1/1PpqP3/4R3/7R/P2B3P/6K1 w - - 3 42";


  static void EnvironmentInit()
  {
    // TODO: Consider using SustainedLowLatency when running under timed time control
    //       However testing with a 10mm node searched did not show much sensitivity to this setting.
    // GCSettings.LatencyMode = GCLatencyMode.Batch;

    //      HardwareManager.Initialize(numaNode);
  }


  // Extensive tests with T75 on HOP show that
  // BS of 1024 is much worse than BS of 512
  // At up to 200,000,000 nodes per game over various TCs we saw:
  //  -23 +/-18
  //  -22 +-20
  //  -27 +/-34
  //  -24 +/-17
  // Test against Lc0 DAG at 120+1 yielded -43 Elo for BS=340 versus -48 Elo for BS=512
  // And: BS 768 vs 512 with T3D at 180+1 on H100 --> 0 +/-18
  //      BS 1024 vs 512  with T3D  at 180+1 on H100 --> 2 +/-18
  // Another test (27 July 2025) with T81 on HOP at 45+0.5s per move vs Lc0 Classic
  //   BS 384  0 +/-12
  //   BS 768 -2 +/-12
  // Another experiment looking at actual batch sizes that make it thru 
  // to neural network in endgames often gets extremely small. In that case,
  // really large batch sizes (1024 to 4096) are helpful for speed.
  // In another test at 45+0.75 on HOP with T3D, BS 384 and BS 768 yielded identical Elo.
  const int MAX_BATCH_SIZE = 1024;

  internal static Ceres.MCTS.Params.ParamsSelect SELECT_PARAMS_MCTS(ParamsSelect paramsSelect)
  {
    if (paramsSelect != default) throw new NotImplementedException("SELECT_PARAMS_MCTS not implemented yet");
    return new MCTS.Params.ParamsSelect();
  }

  internal static Ceres.MCTS.Params.ParamsSearch SEARCH_PARAMS_MCTS(ParamsSearch mcgsParams)
  {
    return new Ceres.MCTS.Params.ParamsSearch() with
    {
#if PLAIN
      TreeReuseEnabled = false,
      Execution = new Ceres.MCTS.Params.ParamsSearchExecution() with
      {
        MaxBatchSize = mcgsParams.Execution.MaxBatchSize
      }
#endif
      EnableInstamoves = mcgsParams.EnableQuickMoves,
      MoveFutilityPruningAggressiveness = mcgsParams.MoveFutilityPruningAggressiveness,
      FutilityPruningStopSearchEnabled = mcgsParams.FutilityPruningStopSearchEnabled,

      EnableSearchExtension = false,
      BestMoveMode = mcgsParams.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN
                                ? MCTS.Params.ParamsSearch.BestMoveModeEnum.TopN
                                : MCTS.Params.ParamsSearch.BestMoveModeEnum.TopQIfSufficientN,
      TreeReuseEnabled = mcgsParams.GraphReuseEnabled,
      ReusePositionEvaluationsFromOtherTree = false,
      EnableTablebases = mcgsParams.EnableTablebases,
      
      Execution = new Ceres.MCTS.Params.ParamsSearchExecution() with
      {
        TranspositionMode = MCTS.Params.TranspositionMode.SingleNodeDeferredCopy,                                                                           

        FlowDirectOverlapped = mcgsParams.Execution.DualOverlappedIterators,
        FlowDualSelectors = mcgsParams.Execution.DualOverlappedIterators,

        SelectParallelEnabled = mcgsParams.Execution.SelectOperationParallelThresholdNumVisits < 999,

        FlowSplitSelects = false, // *** NOTE: The default value of true seems to be worse at 2800 nodes
        MaxBatchSize = mcgsParams.Execution.MaxBatchSize
      }
    };
  }


  // ------------------  COMMON ------------------ 
  public static readonly ParamsSearch SEARCH_PARAMS_MCGS_COMMON = new ParamsSearch() with
  {
//ReusePositionEvaluationsFromOtherTree = true,
    //      BestMoveMode = ParamsSearch.BestMoveModeEnum.TopN,
    //      BestMoveMode = ParamsSearch.BestMoveModeEnum.RegularizedPolicyOptimizationLow,

    BestMoveMode = ParamsSearch.BestMoveModeEnum.TopQIfSufficientN,

    PrefetchParams = PREFETCH_PARAMS,
    EnableGraph = ENABLE_GRAPH,
    DebugDumpVerifyMode = DEBUG_VERIFY,
    MaxNodes = MAX_ALLOCATED_SEARCH_NODES,
    
    // TranspositionStopMinSupportRatio = 2,
    //BackupTranspositionRedescentMinMultipleToStop = 3, // !!!!!!!
    // TestFlag=true,
    //FutilityPruningStopSearchEnabled = false,
    //MoveFutilityPruningAggressiveness = 0,
    // fast, unsure if bugfree  InitFromTranspositionMode = InitFromTPModeEnum.CopyDirectDuringSelect,


    // ************** NOTE: early highly incomplete testing suggests deferred copy may lose Elo **********
    //InitFromTranspositionMode = InitFromTPModeEnum.CopyDirectDuringEvaluate,
    // ********************************************************************************************

    //GraphReuseEnabled = false,

    Execution = new ParamsSearchExecution() with
    {
      MaxBatchSize = MAX_BATCH_SIZE,
      
SelectOperationParallelThresholdNumVisits = SINGLE_THREAD_MODE ? int.MaxValue : new ParamsSearchExecution().SelectOperationParallelThresholdNumVisits,
BackupMode = SINGLE_THREAD_MODE ? BackupMethodEnum.ReductionSingleThread : new ParamsSearchExecution().BackupMode,

DualEvaluators = SINGLE_THREAD_MODE ? false : true,
DualOverlappedIterators = SINGLE_THREAD_MODE ? false : true,
    }
  };


  const bool SINGLE_THREAD_MODE = false;
  const bool USE_COALESCE_MCGS1 = true;
  const bool USE_COALESCE_MCGS2 = true;

  // ------------------ MCGS ------------------ 
  public static readonly ParamsSearch SEARCH_PARAMS_MCGS = SEARCH_PARAMS_MCGS_COMMON with
  {
    //  BestMoveMode = ParamsSearch.BestMoveModeEnum.RegularizedPolicyOptimizationLow,
    // TestFlag for bullet.
    // For settings as checked in overnight on 7/25 to 7/26
    //   - T75 45s+0.5s on DEV vs self MCGS was -6 +/-10, lower nps
    //   - T81 45s+0.5s on HOP vs Lc0DAG was -59Elo (+/-20) at full strength
    //     but  (as above but with lower frequency and higher CPUCT multiplier) was 
//EnableTablebases = false,
//EnableGraph = false,
//GraphReuseEnabled = false,
    //RecomputeNodeStatsDuringSelect = false,
    //EnableSupplementalBackupRecompute=true,
    //EnableGraphCatchUp = true,
    //EnableEarlySmallBatchSizes = true,
    //EnableGraph = false,
//TestFlag = true,
//TestFlag2 = true,
    //MoveOrderingPhase = ParamsSearch.MoveOrderingPhaseEnum.NodeInitializationAndChildSelect,
    //NodeRecalculationPhase = MCGSPhase.Backup,


    EnablePseudoTranspositionBlending =  USE_COALESCE_MCGS1 ? false : true,
    PathTranspositionMode = USE_COALESCE_MCGS1 ? PathMode.PositionEquivalence 
                                               : PathMode.PositionAndHistoryEquivalence,
    //TranspositionStopMinSupportRatio = 2,

    //ValidateAfterSearch = true,    
    //EnablePVAutoExtend = true,
   
    //MoveFutilityPruningAggressiveness = 2,

    // Attempts at using suboptimality were not very successful at longer searches
    // Using 0.03 as the threshold (possibly with a small BS multiplier like 1.3) seems reasonable.
    // But results not good, 10k nodes with T3 is only +15 Elo and much slower due to small BS    

    //    VisitSuboptimalityRejectThreshold = 0.06f,//0.05f,

    //EnableGraph = false,

    //BatchSizeMultiplier = 1.30f,
    //EnableFocusSelection = true,
    //TestFlag = true,

    //OffPathBackupNumAdditionalLevelsToPropagate = 1,
    
    //BackupTranspositionRedescentMinMultipleToStop = 1,//10.0f,
    Execution = SEARCH_PARAMS_MCGS_COMMON.Execution with
    {
      //BackupMode =  BackupMethodEnum.ReductionSingleThread,
      //BackupMode = USE_COALESCE_MCGS1 ? BackupMethodEnum.ReductionSingleThread : BackupMethodEnum.ReductionMultiThread,
      
      // MaxBatchSize = MAX_BATCH_SIZE,

      //SelectOperationParallelThresholdNumVisits = 18,
      //BackupMode = BackupMethodEnum.ReductionSingleThread,
    }
  };


  // ------------------  MCGS2 ------------------ 
  internal static readonly ParamsSearch SEARCH_PARAMS_MCGS2 = SEARCH_PARAMS_MCGS_COMMON with
  {

//    EnableDeferredPolicyCopyFromTransposition = false,
//    InitFromTranspositionMode = InitFromTPModeEnum.CopyDirectDuringEvaluate,

//TestFlag=true,
    //TestFlag2=true,
    //BackupTranspositionRedescentMinMultipleToStop = 1,
 
    EnablePseudoTranspositionBlending = USE_COALESCE_MCGS2 ? false : true,
    PathTranspositionMode = USE_COALESCE_MCGS2 ? PathMode.PositionEquivalence 
                                               : PathMode.PositionAndHistoryEquivalence,

//EnablePathDependentCPUCTScaling = false,
//EnablePseudoTranspositionBlending = false,
    //BackupTranspositionRedescentMinMultipleToStop = 10.0f,

    //BackupFullySynchronizesEdgeWithChild = false,
    Execution = SEARCH_PARAMS_MCGS_COMMON.Execution with
    {
      //BackupMode = USE_COALESCE_MCGS2 ? BackupMethodEnum.ReductionSingleThread : BackupMethodEnum.ReductionMultiThread,
//NNBatchSizeAlignmentTarget = 0,

      //      BackupMode = BackupMethodEnum.LeafToRootSingleThread,
    }
  };



  public static readonly ParamsSelect SELECT_PARAMS_MCGS = new ParamsSelect() with
  {
#if NOT
    OffPolicySelectionFraction = 0.05f,
    OffPolicyMinN = 10,
    OffPolicyTemperatureMultiplier = 1.5f,
    OffPolicyCPUCTMultiplier = 2,
#endif
//RPOBackupLambda = 1.5f,// 1.75f,
// 0.75 --> -5 @2000
//CPUCT = new ParamsSelect().CPUCT * 0.90f,
    //PolicySoftmax = 1.0f,

    // turning off RPOSelect (use only RPOBackup) solid at small searches but -50 Elo at 5000 nodes/move
    //      RPOSelectLambda = ENGINE1_USE_RPO ? 0 * 0.9f : 0, // bad?
    //    RPOBackupLambda = ENGINE1_USE_RPO ? 0.6f : 0, // not so bad, at 0.6 with MinN=10?
    //    RPOLambdaPower = ENGINE1_USE_RPO ? 0.5f : 0,
    //    RPOBackupMinN = ENGINE1_USE_RPO ? 10 : 0

#if NOT
    CPUCT = 2.897f,
    CPUCTFactor = 3.973f,
    CPUCTBase = 45569,

    PolicySoftmax = 1.4f,
    FPUValue = 0.984f
#endif
  };

  internal static readonly ParamsSelect SELECT_PARAMS_MCGS2 = new ParamsSelect() with
  {
//RPOBackupLambda = 1.5f,// 1.75f,

#if NOT
    CPUCT = 2.897f,
    CPUCTFactor = 3.973f,
    CPUCTBase = 45569,
    CPUCTAtRoot = 2.897f,
    CPUCTFactorAtRoot = 3.973f,
    CPUCTBaseAtRoot = 45569,

    PolicySoftmax = 1.4f,
    FPUValue = 0.984f
#endif
    //      RPOSelectLambda = ENGINE2_USE_RPO ? 0.9f : 0,
    //      RPOBackupLambda = ENGINE2_USE_RPO ? 0.6f : 0,
    //      RPOLambdaPower = ENGINE2_USE_RPO ? 0.5f : 0,
  };



  const bool LEPNED_BENCHMARK = true;
  const bool DISPOSE_AFTER_SEARCH = false;
  const bool ENABLE_GRAPH = true; 
  const bool DEBUG_VERIFY = false;
  const int TEST_SEARCH_NODES = LEPNED_BENCHMARK ? 10_000_000 : 125_500_000; // must be less than MAX_ALLOCATED_SEARCH_NODES
  const int MAX_ALLOCATED_SEARCH_NODES = TEST_SEARCH_NODES;
  const int NUM_ITER_WARMUP = 10;
  const bool PREFETCH = false;
  const bool PREFETCH_REARRANGE = false;

  //      int[] NODES_PER_LEVEL = BIG ? new int[] { 1, 12, 50, 200 } : new int[] { 1, 10, 35 };//,  14 };//, 70, 300, 700};
  //      int[] NODES_PER_LEVEL = BIG ? new int[] { 1, 15, 100, 400 } : new int[] { 1, 10, 35 };//,  14 };//, 70, 300, 700};
  /*
      static readonly ParamsPrefetch PREFETCH_PARAMS_1000 = !PREFETCH ? null : new ParamsPrefetch() with
      {
        NumDepthLevels = 4,
        MaxNumNodes = 999,

        MaxNodesPerDepth = [1, 50, 500, 1000],
        MaxWidth = int.MaxValue,

        MinAbsolutePolicyPctPerDepth = [0, 0, 2, 10],// good: [1, 1, 2, 10];
        MaxProbabilityPctGapFromBestPerDepth =  null,//new float[] { 100, 50, 30, 20 },
      };
  */
  static ParamsPrefetch Prefetch(float[] minAbsPolicy)
    => new ParamsPrefetch() with
    {
      NumDepthLevels = minAbsPolicy.Length,
      MinAbsolutePolicyPctPerDepth = minAbsPolicy,
      PrefetchResortChildrenUsingV = PREFETCH_REARRANGE
    };

  static readonly ParamsPrefetch PREFETCH_PARAMS_1000 = Prefetch([0, 0, 2, 10]);
  static readonly ParamsPrefetch PREFETCH_PARAMS_100 = Prefetch([0, 1, 15, 25]);
  static readonly ParamsPrefetch PREFETCH_PARAMS_200 = Prefetch([0, 1, 10, 20]);
  static readonly ParamsPrefetch PREFETCH_PARAMS_10000 = Prefetch([0, 0, 2, 10, 20]);

  static readonly ParamsPrefetch PREFETCH_PARAMS_ALL_DEPTH_3 = Prefetch([0, 0, 0]);
  static readonly ParamsPrefetch PREFETCH_PARAMS_ALL_DEPTH_4 = Prefetch([0, 0, 0, 0]);

  static readonly ParamsPrefetch PREFETCH_PARAMS = !PREFETCH ? null : PREFETCH_PARAMS_200;




  public static void ConvertTest()
  {
    MGPosition mgPos = Position.StartPosition.ToMGPosition;
    EncodedMove encodedMove = EncodedMove.FromNeuralNetIndex(33); // some random move
    NNEvaluatorResult r = NNEvaluator.FromSpecification("~T70", "GPU:0").Evaluate(mgPos.ToPosition);
    EncodedMove x = r.Policy.PolicyInfoAtIndex(0).Move;

    // 86mm/sec
    while (true)
    Benchmarking.DumpOperationTimeAndMemoryStats(() => ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in mgPos), "conv");

  }


  static void CompareDistrib(GameEngineSearchResultCeresMCGS mcgs, GameEngineSearchResultCeres mcts)
  {
    int childIndex = 0;
    foreach (GEdge edgeMCGS in mcgs.Search.SearchRootNode.ChildEdgesExpanded)
    {
      GNode nodeMCGS = edgeMCGS.ChildNode;
      MCTSNode nodeMCTS = mcts.Search.SearchRootNode.ChildAtIndex(childIndex);

      Console.WriteLine($"{edgeMCGS.Move} {edgeMCGS.N} {nodeMCGS.N}  {edgeMCGS.Q,6:F2}  {nodeMCTS.Q,6:F2}  ");

      Console.WriteLine();
      childIndex++;
    }
  }

  static void CompareDistrib(GameEngineSearchResultCeresMCGS mcgs1, GameEngineSearchResultCeresMCGS mcgs2)
  {
    float lambdaPower = mcgs1.Search.Manager.ParamsSelect.RPOLambdaPower;
    int numChildrenExpanded = mcgs1.Search.SearchRootNode.NumEdgesExpanded;
    RPOResult rpoResultSelection = RPOTests.BestMoveInfo(mcgs1.Search.SearchRootNode,  float.NaN, numChildrenExpanded, mcgs1.Search.Manager.ParamsSelect.RPOSelectLambda, lambdaPower);
    RPOResult rpoResultBackup = RPOTests.BestMoveInfo(mcgs1.Search.SearchRootNode, float.NaN, numChildrenExpanded, mcgs1.Search.Manager.ParamsSelect.RPOBackupLambda, lambdaPower);

    Console.WriteLine("  Move      RPO N      PUCT N        Pol    Emp    Sel    Bck      PUCT       QDiff");
    int childIndex = 0;
    float mcgs1N = mcgs1.Search.SearchRootNode.N;
    float mcgs2N = mcgs2.Search.SearchRootNode.N;
    foreach (GEdge edgeMCGS1 in mcgs1.Search.SearchRootNode.ChildEdgesExpanded)
    {
      if (mcgs2.Search.SearchRootNode.NumEdgesExpanded > childIndex
        && rpoResultBackup.optimalP?.Length > childIndex)
      {
        GEdge edgeMCGS2 = mcgs2.Search.SearchRootNode.ChildEdgeAtIndex(childIndex);

        double diffQ = edgeMCGS1.Q - edgeMCGS2.Q;
        float policy = edgeMCGS1.P;
        float policyMCGS2 = edgeMCGS2.N / mcgs2N;
        float policyEmp = edgeMCGS1.N / mcgs1N;
        double policySelect = rpoResultSelection.optimalP[childIndex];
        double policyBackup = rpoResultBackup.optimalP[childIndex];
        Console.WriteLine($"{edgeMCGS1.Move,6}  {edgeMCGS1.N,10:N0}  {edgeMCGS2.N,10:N0}    "
            + $"{(policy < 0.005f ? "".PadLeft(7) : $"{100 * policy,7:F1}")}"
            + $"{(policyEmp < 0.005f ? "".PadLeft(7) : $"{100 * policyEmp,7:F1}")}"
            + $"{(policySelect < 0.005f ? "".PadLeft(7) : $"{100 * policySelect,7:F1}")}"
            + $"{(policyBackup < 0.005f ? "".PadLeft(7) : $"{100 * policyBackup,7:F1}")}    "
            + $"{(policyMCGS2 < 0.005f ? "".PadLeft(7) : $"{100 * policyMCGS2,7:F1}")}    "
            + $"{diffQ,6:F2}    {edgeMCGS1.Q,6:F2}  {edgeMCGS2.Q,6:F2}  ");
        childIndex++;
      }
    }
  }

  public static void DumpForChild(GParentsStore table, int parentIndex)
  {
    Console.WriteLine("Dumping parents for entry " + parentIndex);
    Span<int> parents = stackalloc int[50];
    table.GetParentsNodeIndices(new NodeIndex(parentIndex), parents);
    for (int i = 0; i < parents.Length && parents[i] != -1; i++)
    {
      Console.WriteLine($"  Parent for {parentIndex} = {parents[i]}");
    }
  }

  static bool SILENT = false;

  public class ComparatorMCGSvsMCTS
  {
    public GameEngineCeresMCGSInProcess gameEngineCeresMCGS;
    public GameEngineCeresMCGSInProcess gameEngineCeresMCGS2;
    public GameEngineCeresInProcess gameEngineCeresMCTS;
    readonly GameEngineLC0 gameEngineLc0;

    public ComparatorMCGSvsMCTS()
    {
      gameEngineLc0 = RUN_LC0 ? (GameEngineLC0)GameEngineLc0(NET_LC0, DEVICE, LC0EngineType.RewriteDAG, false, true).CreateEngine() : null;
      
      NNEvaluatorDef defNet = NNEvaluatorDef.FromSpecification(NET_CERES, DEVICE);

      GameEngineDefCeresMCGS gameEngineDefCeresMCGS = new("CeresMCGS1", defNet, SEARCH_PARAMS_MCGS, SELECT_PARAMS_MCGS);
//        GameEngineDefCeresMCGS gameEngineDefCeresMCGS = new("CeresMCGS1", defNet, SEARCH_PARAMS_MCGS, SELECT_PARAMS_MCGS);
      gameEngineCeresMCGS = gameEngineDefCeresMCGS.CreateEngine() as GameEngineCeresMCGSInProcess;
      gameEngineCeresMCGS.DisposeGraphAfterSearch = false;

      if (!RUN_LC0)
      {
        GameEngineDefCeresMCGS gameEngineDefCeresMCGS2 = new("CeresMCGS2", defNet, SEARCH_PARAMS_MCGS2, SELECT_PARAMS_MCGS2);
        gameEngineCeresMCGS2 = gameEngineDefCeresMCGS2.CreateEngine() as GameEngineCeresMCGSInProcess;
        gameEngineCeresMCGS2.DisposeGraphAfterSearch = false;

        GameEngineDefCeres gameEngineDefCeresMCTS = new("CeresMCGS1", defNet, default,
                                                        SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS), new MCTS.Params.ParamsSelect());
        gameEngineCeresMCTS = gameEngineDefCeresMCTS.CreateEngine() as GameEngineCeresInProcess;
      }
    }


//    const string NET_CERES = "~T79";
//    const string NET_LC0 = "~T79";

    const string NET_CERES = "~T1_DISTILL_256_10_FP16_TRT|cudagraphs=false";
    const string NET_LC0 = "~T1_DISTILL_256_10_NATIVE";

    const string DEVICE = "GPU:0";
    const bool RUN_MCGS2 = false;
    const bool RUN_LC0 = true;
    
    public (int, int) RunTest(int numPositions = int.MaxValue)
    {
      int correctMCGS1 = 0;
      int correctMCGS2 = 0;
      List<float> diffs = [];

      const int NODE_LIMIT = 10_000;// 145; 
      const int SKIP_COUNT = 73;
      const float Q_DELTA_THRESHOLD = 0.10f; // <-------------- THRESHOLD
      const bool ALWAYS_DUMP = false;

      SearchLimit limit = SearchLimit.NodesPerMove(NODE_LIMIT);
      //limit = SearchLimit.SecondsPerMove(0.5f);

      const bool SINGLE_POS_TEST = false;
      if (SINGLE_POS_TEST)
      {
        const string CERES_DRAWISH = null;// "8/8/6p1/1kB3b1/1P6/1K3P2/8/8 w - - 4 52";
        PositionWithHistory singleTest = PositionWithHistory.FromFENAndMovesSAN(CERES_DRAWISH);
        SearchLimit limitSingle = SearchLimit.NodesPerMove(950);

        GameEngineSearchResultCeresMCGS resultCeresMCGS = gameEngineCeresMCGS.Search(singleTest, limitSingle) as GameEngineSearchResultCeresMCGS;
        Console.WriteLine(resultCeresMCGS.Engine.Graph.GraphRootNode.NodeRef.Q);

        GameEngineSearchResultCeres resultCeresMCTS = gameEngineCeresMCTS.Search(singleTest, limitSingle) as GameEngineSearchResultCeres;
        Console.WriteLine(resultCeresMCTS.ScoreQ);

        System.Environment.Exit(3);
      }


      string FN = SoftwareManager.IsLinux ? @"/mnt/devd/tar/training-run1-test80-20240531-1317.tar"
                                          : @"d:\tar\training-run1-test80-20240531-1317.tar";
      const long MAX_POSITIONS = 5_000 * SKIP_COUNT; 
      IEnumerable<PositionWithHistory> positions = new EncodedTrainingPositionReaderTAR(FN)
                                                                                        .EnumeratePositions(maxPositions: MAX_POSITIONS).Select(p => p.ToPositionWithHistory());

      int count = 0;
      int countAll = 0;
      foreach (PositionWithHistory loopTestPos in positions)
      {
        if (count > numPositions)
        {
          DumpDiffs(diffs, count);
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "CORRECT 1/2: " + correctMCGS1 + " " + correctMCGS2);
          return (correctMCGS1, correctMCGS2);
        }

        if (countAll++ % SKIP_COUNT != 0)
        {
          continue;
        }

        gameEngineCeresMCTS?.ResetGame();
        gameEngineCeresMCGS?.ResetGame();
        gameEngineCeresMCGS2?.ResetGame();
        gameEngineLc0?.ResetGame();

        PositionWithHistory testPos = loopTestPos;

        if (testPos.FinalPosition.CalcTerminalStatus() != GameResult.Unknown
          || testPos.FinalPosition.PieceCount <= 7
          )
        {
          //          continue;
        }

        // Non-null value causes only that position to be tested
        const string SINGLE_FEN_TO_TEST = null;// "2N5/6bk/5pp1/1b4Bp/p2pP1nP/P4BP1/1P3P2/6K1 w - - 0 1";// "k7/5Q2/p2pR3/8/P1P5/1P4KP/8/8 b - c3 0 1";// "1K2q3/2B5/1P6/2k1p3/4P3/1B6/8/8 w - - 23 1";
        if (SINGLE_FEN_TO_TEST != null && SINGLE_FEN_TO_TEST != testPos.FinalPosition.FEN)
        {
            //testPos = PositionWithHistory.FromFENAndMovesUCI(SINGLE_FEN_TO_TEST);
              continue;
        }

        if (++count % 50 == 0 && !SILENT)
        {
          DumpDiffs(diffs, count);
        }

        GameEngineSearchResultCeresMCGS resultCeresMCGS = gameEngineCeresMCGS.Search(testPos, limit) as GameEngineSearchResultCeresMCGS;

        const bool DUMP_Q = false;
        if (DUMP_Q)
        {
#if NOT
          //          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, Math.Round(resultCeresMCGS.ScoreQ, 2).ToString() + " " + testPos.FinalPosition.FEN);

          Graph graph = resultCeresMCGS.Engine.Graph;
          for (int i = graph.Store.NodesStore.NumUsedNodes - 2; i > 0; i--)
          {
            GNode node = graph[i];
            double diff = node.Q - node.ComputeQPure();
            if (Math.Abs(diff) > 0.03)
            {
              Console.WriteLine(diff + "   "  +  $"Node #{i} N={node.N} SibFrac={node.NodeRef.SiblingsQFrac} Q={node.Q,6:F3} QPure={node.ComputeQPure(),6:F3}" 
                               + $" {node.CalcPosition().ToPosition.FEN}");
            }
          }
#endif
          Console.WriteLine();
        }

        if (false)
        {
          const int MIN_VISITS = 20;
          Console.WriteLine();
          throw new NotImplementedException();
//          MCGSTest.AnalyzeSearchTreeTerminalPlayouts(gameEngineCeresMCGS.Search, MIN_VISITS);
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, Math.Round(resultCeresMCGS.ScoreQ, 2).ToString() + " " + testPos.FinalPosition.FEN);
          Console.WriteLine();
          //continue;
        }

        if (RUN_VALIDATION)
        {
Console.WriteLine(testPos.FinalPosition.FEN);
          resultCeresMCGS.Engine.Graph.Validate(true);
          //resultCeresMCGS.Engine.Graph.DumpNodesStructure(filterNodeIndex: new NodeIndex(14));
        }

        if (RUN_LC0)
        {
          GameEngineSearchResult resultLc0 = gameEngineLc0.Search(testPos, limit);

          float delta = Math.Abs(resultCeresMCGS.ScoreQRoot - resultLc0.ScoreQ);
          diffs.Add(delta);

          bool sameMove = resultCeresMCGS.MoveString == resultLc0.MoveString;

          if (ALWAYS_DUMP || delta >= Q_DELTA_THRESHOLD)
          {
            if (!SILENT)
            {
              Console.WriteLine();
              Console.WriteLine(resultCeresMCGS.ScoreQRoot + " " + resultLc0.ScoreQ + " " + testPos);
              Console.WriteLine(resultCeresMCGS.MoveString + " " + resultLc0.MoveString);
              gameEngineCeresMCGS.Search.Manager.DumpFullInfo(resultCeresMCGS, Console.Out, testPos.FENAndMovesString);
              foreach (VerboseMoveStat ss in resultLc0.VerboseMoveStats)
              {
                Console.WriteLine(ss);
              }
              Console.WriteLine();

              Console.WriteLine("DELTA: " + delta + "   same move: " + sameMove);
            }
          }
          else
          {
            Console.Write(".");
          }
        }
        else if (RUN_MCGS2)
        {
          const float MCGS2_TIME_SCALE_FACTOR = 1f;
          GameEngineSearchResultCeresMCGS resultCeresMCGS2 = gameEngineCeresMCGS2.Search(testPos, limit * MCGS2_TIME_SCALE_FACTOR) as GameEngineSearchResultCeresMCGS;
          if (RUN_VALIDATION)
          {
            resultCeresMCGS2.Engine.Graph.Validate(true);
          }

          float delta = Math.Abs(resultCeresMCGS.ScoreQRoot - resultCeresMCGS2.ScoreQRoot);
          diffs.Add(delta);

          bool sameMove = resultCeresMCGS.MoveString == resultCeresMCGS2.MoveString;

          if (ALWAYS_DUMP || delta >= Q_DELTA_THRESHOLD)
          {
            if (!SILENT)
            {
              Console.WriteLine();
              Console.WriteLine(resultCeresMCGS.ScoreQRoot + " " + resultCeresMCGS2.ScoreQRoot + " " + testPos);
              Console.WriteLine(resultCeresMCGS.MoveString + " " + resultCeresMCGS2.MoveString);
              //Console.WriteLine("Num tablebase hits        : " + EvaluatorSyzygy.NumHits.Value);
              Console.WriteLine();

              int MAX_DEPTH = resultCeresMCGS.FinalN <= 40 ? int.MaxValue : 2;
              if (MAX_DEPTH < int.MaxValue || resultCeresMCGS.Search.SearchRootNode.N < 50)
              {
                resultCeresMCGS.Engine.Graph.DumpNodesStructure(maxDepth: MAX_DEPTH);
                Console.WriteLine("\r\n");
                resultCeresMCGS2.Engine.Graph.DumpNodesStructure(maxDepth: MAX_DEPTH);
              }

              Console.WriteLine("DELTA: " + delta + "   same move: " + sameMove);

              CompareDistrib(resultCeresMCGS, resultCeresMCGS2);
            }

            const bool FULL_DEBUG = false;
            if (FULL_DEBUG && !sameMove)
            {
              const float LIMIT_MULTIPLIER = 4;
              GameEngineSearchResultCeres resultCeresMCTS = gameEngineCeresMCTS.Search(testPos, limit * LIMIT_MULTIPLIER) as GameEngineSearchResultCeres;

              if (!SILENT)
              {
                Console.WriteLine("\r\nMCTS BIG DUMP");
                resultCeresMCTS.Search.Manager.DumpFullInfo(Console.Out, testPos.FENAndMovesString);
                Console.WriteLine();
              }
              bool mcgs1Correct = resultCeresMCGS.MoveString == resultCeresMCTS.MoveString;
              bool mcgs2Correct = resultCeresMCGS2.MoveString == resultCeresMCTS.MoveString;

              if (mcgs1Correct) correctMCGS1++;
              if (mcgs2Correct) correctMCGS2++;
            }
          }
          else
          {
            Console.Write(".");
          }
          resultCeresMCGS2.Engine.Graph.Store.Dispose();
        }
        else
        {
          GameEngineSearchResultCeres resultCeresMCTS = gameEngineCeresMCTS.Search(testPos, limit) as GameEngineSearchResultCeres;

          //          float qDiff = (float)Math.Abs((resultCeresMCTS.Search.SearchRootNode.Q - resultCeresMCGS.ScoreQRoot));
          float qDiff = (float)Math.Abs(resultCeresMCTS.Search.SearchRootNode.Q - resultCeresMCGS.ScoreQRoot);
          diffs.Add(qDiff);

          bool sameMove = resultCeresMCTS.MoveString == resultCeresMCGS.MoveString;

          bool looksLikeDraw = Math.Abs(resultCeresMCTS.Search.SearchRootNode.Q) < 0.01;
          bool looksLikeTerminal = looksLikeDraw || MathF.Abs(resultCeresMCGS.ScoreQRoot) > 0.99;
          int numLegalMoves = resultCeresMCTS.Search.SearchRootNode.NumPolicyMoves;

          if (true //numLegalMoves > 1
            && (ALWAYS_DUMP
               // too delicate              || !sameMove
               || qDiff > Q_DELTA_THRESHOLD
               || (!sameMove && !looksLikeTerminal && qDiff > Q_DELTA_THRESHOLD / 2)))
          {
            Console.WriteLine();
            Console.WriteLine("---------------------------------------------------------------------------------------------------------");
            Console.WriteLine("Position " + testPos);
            Console.WriteLine("Best moves MCGS/MCTS " + resultCeresMCGS.MoveString + " " + resultCeresMCTS.MoveString);
            if (resultCeresMCTS.Search.Manager.Context.Tree.Root.N < 50)
            {
              resultCeresMCTS.Search.Manager.Context.Tree.Store.Dump(NODE_LIMIT < 5);
            }
            resultCeresMCTS.Search.Manager.DumpFullInfo(Console.Out, testPos.FENAndMovesString);
            Console.WriteLine();
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "MCTS " + resultCeresMCTS.Search.SearchRootNode.Q + " " + resultCeresMCTS.MoveString);
            Console.WriteLine("---------------------------------------------------------------------------------------------------------");
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "MCGS " + resultCeresMCGS.ScoreQRoot + " " + resultCeresMCGS.MoveString);
            Console.WriteLine();
            if (RUN_VALIDATION)
            {
              resultCeresMCGS.Engine.Graph.Validate();
            }

            if (resultCeresMCGS.Search.SearchRootNode.N < 50)
            {
              resultCeresMCGS.Engine.Graph.DumpNodesStructure();
            }

            resultCeresMCGS.Search.Manager.DumpFullInfo(resultCeresMCGS, Console.Out, testPos.FENAndMovesString);

            Console.WriteLine();
            Console.WriteLine();
            DumpNNStats();
          }
          else
          {
            if (qDiff > 0.05)
            {
              Console.WriteLine();
              Console.WriteLine("draw? " + looksLikeDraw + "  " + Math.Round(qDiff, 3) + " " + " match " + resultCeresMCTS.Search.SearchRootNode.Q + " " + resultCeresMCTS.MoveString + " " + testPos);
            }
            else
            {
              Console.Write(".");
            }
          }
        }

        if (SINGLE_FEN_TO_TEST != null)
        {
//            throw new NotImplementedException();
        }
      }
      return default;
    }
  }

  public static (int, int) CompareMCGSvsMCTS(int numPositions = int.MaxValue)
  {
    ComparatorMCGSvsMCTS comparator = new();
    return comparator.RunTest(numPositions);
  }


  private static void DumpDiffs(List<float> diffs, int count)
  {
    if (diffs.Count > 0)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"\r\nStatistics of diffs ({count})");
      Console.WriteLine("avg abs     : " + diffs.Average(x => Math.Abs(x)));
      Console.WriteLine("max abs     : " + diffs.Max(x => Math.Abs(x)));
    }
  }


#if NOT
    // Certainty propagation  test
    if (false)
    {
      // ======================================================================================================================================================
      string FEN_BLACK_FORCED_MOVE_WHITE_MULTIPLE_MATES = "7k/4B1p1/6B1/7Q/q7/8/r5r1/4K3 b - - 0 1";
  string FEN_TEST = "8/6p1/8/6k1/3P1NP1/1n2K3/n7/4B3 w - - 1 55"; // white winning
  PositionWithHistory pwhTest = PositionWithHistory.FromFENAndMovesUCI(FEN_TEST, "");

  var testSearchResult = engineMCGS1.CreateEngine().Search(pwhTest, SearchLimit.NodesPerMove(5));
  Console.WriteLine(testSearchResult);
      System.Environment.Exit(3);
      //    internal static Func<(GameEngineCeresMCGSInProcess engine, PositionWithHistory Pos, SearchLimit Limit), (MGMove, float, int)> MoveMaker =
      // ======================================================================================================================================================
    }
#endif


  public static void SinglePositionMiscTests()
  {
    //      string TEST_POSITION_TO_USE = "1brk2r1/p2nnqp1/1pp1p2p/3p1p2/2PP1P1P/1P2PN2/PB2KP2/1BQR3R w - - 4 13";

    string TEST_POSITION_TO_USE =  "8/1q5k/6p1/K2p2Pp/1P1Q3P/8/8/8 b - - 1 57";// FEN_UNRECOGNIZED_ENDGAME_DRAW;// FEN_H2H3_CERES_MCGS_SLOWER_THAN_LC0_DAG; //FEN_TCEC_EMBARRASING_ENDGAME_LOSS;// 
                                                                               //    TEST_POSITION_TO_USE = Position.StartPosition.FEN;
    //string NET_ID = "C1-640-34-i8";
    //    NET_ID = "~T3_DISTILL_512_15_FP16_TRT";
    string NET_ID = "~T75";
//NET_ID = "badgyal-3.pb.gz";
    //string NET_ID = "~T1_DISTILL_256_10_FP16_TRT|cudagraphs=true";
    const string DEVICE = "GPU:0#TensorRT16";

    GameEngineCeresMCGSInProcess engineMCGS = new("x", NNEvaluatorDef.FromSpecification(NET_ID, DEVICE),
                                                  disposeGraphAfterSearch: DISPOSE_AFTER_SEARCH,
                                                  searchParams: SEARCH_PARAMS_MCGS,
                                                  selectParams: SELECT_PARAMS_MCGS);

    if (false)
    {
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(engineMCGS.SearchParams, new ParamsSearch()), false);
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(engineMCGS.SearchParams.Execution, new ParamsSearch().Execution), false);
      Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(engineMCGS.SelectParams, new ParamsSelect()), false);
    }

    PositionWithHistory positionToTest = PositionWithHistory.FromFENAndMovesSAN(TEST_POSITION_TO_USE);

    const int WARMUP_N = 5_000;
    GameEngineSearchResultCeresMCGS mcgsResult = engineMCGS.Search(positionToTest, SearchLimit.NodesPerMove(WARMUP_N)) as GameEngineSearchResultCeresMCGS; /// warmup
    if (RUN_VALIDATION)
    {
      mcgsResult?.Engine.Graph.Validate();
    }

    // Warmup and speed test
    using (new TimingBlock("MCGS warmup"))
    {
      for (int i = 0; i < NUM_ITER_WARMUP; i++)
      {
        const int WARMUP_NODES = 100_000;
        long startMem = GC.GetTotalAllocatedBytes();
        mcgsResult = engineMCGS.Search(positionToTest, SearchLimit.NodesPerMove(WARMUP_NODES)) as GameEngineSearchResultCeresMCGS;
        long endMem = GC.GetTotalAllocatedBytes();
        long usedMB = (endMem - startMem) / (1024 * 1024);
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Total MB allocated {usedMB}, {GC.GetGCMemoryInfo().PauseTimePercentage}% GC time (size {WARMUP_NODES})");

        if (i < NUM_ITER_WARMUP - 1)
        {
          mcgsResult.Engine.Graph.Store.Dispose();
        }
        engineMCGS.ResetGame();
      }
    }

    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Total kbytes allocated {GC.GetTotalAllocatedBytes() / 1024}, {GC.GetGCMemoryInfo().PauseTimePercentage}% GC time");
    Console.WriteLine();

    GameEngineCeresInProcess engineMCTS;
    using (new TimingBlock("MCTS warmup"))
    {
      engineMCTS = new("x", NNEvaluatorDef.FromSpecification(NET_ID, DEVICE), null,
                       SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS),
                       new Ceres.MCTS.Params.ParamsSelect());

      for (int i = 0; i < NUM_ITER_WARMUP; i++)
      {
        engineMCTS.SearchCeres(positionToTest, SearchLimit.NodesPerMove(20_000)); /// warmup
        engineMCTS.ResetGame();
      }
    }


    GraphPrefetcher.TotalNumRearranged = 0;
    GraphPrefetcher.TotalNumPrefetched = 0;
    ResetNNStats();

    engineMCGS.ResetGame();
    float bestTimeMCGS = float.MaxValue;
    using (new TimingBlock("MCGS big 2x"))
    {
      mcgsResult = engineMCGS.Search(positionToTest, SearchLimit.NodesPerMove(TEST_SEARCH_NODES)) as GameEngineSearchResultCeresMCGS;
      bestTimeMCGS = (float)mcgsResult.TimingStats.ElapsedTimeSecs;
      Console.WriteLine("MCGS 1: " + engineMCGS.UCIInfo.RawString);
      engineMCGS.ResetGame();
      mcgsResult = engineMCGS.Search(positionToTest, SearchLimit.NodesPerMove(TEST_SEARCH_NODES)) as GameEngineSearchResultCeresMCGS;
      bestTimeMCGS = MathF.Min((float)mcgsResult.TimingStats.ElapsedTimeSecs, bestTimeMCGS);
      Console.WriteLine("MCGS 2: " + engineMCGS.UCIInfo.RawString);
    }

    Console.WriteLine();
    Console.WriteLine($"MCGS nps: {engineMCGS.Search.Manager.Engine.SearchRootNode.N / bestTimeMCGS:N0} nodes/sec");
    Console.WriteLine("Memory stats for search to N of " + engineMCGS.Search.Manager.Engine.SearchRootNode.N);
    engineMCGS.DumpStoreUsageSummary();
//    System.Environment.Exit(3);
    Console.WriteLine();

    //      Console.WriteLine("TICKS " + MCGSEvaluatorNeuralNet.timerSetBatch.TotalTicks);

    if (RUN_VALIDATION)
    {
      mcgsResult?.Engine.Graph.Validate();
    }



    GameEngineSearchResultCeres mctsResult;
    engineMCTS.ResetGame();
    float bestTimeMCTS = float.MaxValue;
    using (new TimingBlock("MCTS big 2x"))
    {
      mctsResult = engineMCTS.SearchCeres(positionToTest, SearchLimit.NodesPerMove(TEST_SEARCH_NODES));
      bestTimeMCTS = (float)mctsResult.TimingStats.ElapsedTimeSecs;
      Console.WriteLine("MCTS 1: " + engineMCTS.UCIInfo.RawString);
      engineMCTS.ResetGame();
      mctsResult = engineMCTS.SearchCeres(positionToTest, SearchLimit.NodesPerMove(TEST_SEARCH_NODES));
      bestTimeMCTS = MathF.Min((float)mctsResult.TimingStats.ElapsedTimeSecs, bestTimeMCGS);
      Console.WriteLine("MCTS 2: " + engineMCTS.UCIInfo.RawString);
    }

    Console.WriteLine();
    string hasGraphStr = mcgsResult.Search.Manager.ParamsSearch.EnableGraph ? " (with graph)" : " (without graph)";
    Console.WriteLine("MCGS/MCTS search of size " + TEST_SEARCH_NODES + " nodes from position " + TEST_POSITION_TO_USE + hasGraphStr);
    Console.WriteLine($"MCGS  {mcgsResult.ScoreCentipawns,6:F2} cp   {Math.Round(mcgsResult.TimingStats.ElapsedTimeSecs, 3)} sec  {mcgsResult.FinalN} nodes  {mcgsResult.BestMoveInfo}");
    Console.WriteLine($"MCTS  {mctsResult.ScoreCentipawns,6:F2} cp   {Math.Round(mctsResult.TimingStats.ElapsedTimeSecs, 3)} sec  {mctsResult.FinalN} nodes  {mctsResult.BestMove}");
    Console.WriteLine("Total prefetched: " + GraphPrefetcher.TotalNumPrefetched + " " + GraphPrefetcher.TotalNumRearranged);
    Console.WriteLine();
    Console.WriteLine($"Final graph nodes   : {mcgsResult.Engine.Graph.GraphRootNode.N,12:N0}");
    Console.WriteLine($"Transposition nodes: {mcgsResult.Engine.Graph.NumLinksToExistingNodes,12:N0}");

    Console.WriteLine();
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Total kbytes allocated {GC.GetTotalAllocatedBytes() / 1024}, {GC.GetGCMemoryInfo().PauseTimePercentage}% GC time");

    DumpNNStats();


    System.Environment.Exit(3);

    const bool SPEED_TEST_ENUMERATE_EDGES = true;
    if (SPEED_TEST_ENUMERATE_EDGES)
    {
      MCGSTestPerformance.SpeedTestEnumerateEdges(mcgsResult);
    }
    Console.WriteLine(mcgsResult);
  }


#if NOT
    if (false)
    {
      for (int i = 0; i < 1; i++)
      {
        Console.WriteLine();
        GraphTestingIntegration.ENABLE_PREFETCH = false;
        SinglePositionMiscTests();

        Console.WriteLine();
        GraphTestingIntegration.ENABLE_PREFETCH = true;
        GraphPrefetcher.TotalNumPrefetched = 0;
        SinglePositionMiscTests();

        Console.WriteLine("NUM PREFETCHED: " + GraphPrefetcher.TotalNumPrefetched);
      }
      System.Environment.Exit(3);
    }
#endif


  public static (BestMoveInfoMCGS bestMoveInfo, bool foundClose) 
    ChooseBestMoveIncludingTerminalPlayouts(BestMoveInfoMCGS priorBestMoveInfo, MCGSManager search, int minVisits)
  {
    GEdge recalculatedBestMoveEdge = priorBestMoveInfo.BestMoveEdge;
    bool foundClose = false;
    float bestPlayoutQOfClose = float.MinValue;
    foreach (GEdge edge in search.Engine.SearchRootNode.ChildEdgesExpanded)
    {
      const double THRESHOLD_CLOSE = 0.005f;

      if (edge != priorBestMoveInfo.BestMoveEdge)
      {
        if (Math.Abs(edge.Q - priorBestMoveInfo.BestQEdge.Q) < THRESHOLD_CLOSE
         && edge.N >= priorBestMoveInfo.BestMoveEdge.ChildNode.N / 3)
        {
          MGPosition position = search.RootMGPos;
          MGMove thisMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(edge.Move, position);
          position.MakeMove(thisMove);
          foundClose = true;
          double thisQ = AnalyzeSearchTreeTerminalPlayouts(search.EvaluatorNN0.Evaluator, position, edge.ChildNode, minVisits, false);
          if (thisQ > bestPlayoutQOfClose)
          {
            recalculatedBestMoveEdge = edge;
            bestPlayoutQOfClose = (float)thisQ;
          }
        }
      }
    }

    if (foundClose)
    {
      MGPosition position = search.RootMGPos;
      MGMove thisMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(priorBestMoveInfo.BestMoveEdge.Move, position);
      position.MakeMove(thisMove);
      double priorBestMovePlayoutQ = AnalyzeSearchTreeTerminalPlayouts(search.EvaluatorNN0.Evaluator, 
        position, priorBestMoveInfo.BestMoveEdge.ChildNode, minVisits, false);

      MGMove recalculatedMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(recalculatedBestMoveEdge.Move,
                                                                                    search.Engine.SearchRootNode.CalcPosition());

      if (bestPlayoutQOfClose > priorBestMovePlayoutQ)
      {
          // TODO: set some other fields too? new BestMoveInfoMCGS(BestMoveReason reason, MGPosition parentPos, GEdge bestMoveEdge,
          //                float qMaximal, float bestN, float bestNSecond,
          //                GEdge bestNEdge = default, GEdge bestQEdge = default)

        return (new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, recalculatedMove, (float)recalculatedBestMoveEdge.Q), true);
      }
    }

    return (priorBestMoveInfo, foundClose);
  }


  public static double AnalyzeSearchTreeTerminalPlayouts(MCGSSearch search, MGPosition rootPos, int minVisits, bool verbose = false)
  {
    return AnalyzeSearchTreeTerminalPlayouts(search.Manager.EvaluatorsSet.Evaluator0, rootPos, search.SearchRootNode, minVisits, verbose);
  }

  public static double AnalyzeSearchTreeTerminalPlayouts(NNEvaluator evaluator, MGPosition rootPos, GNode rootNode, int minVisits, bool verbose)
  {
    PrincipalPosSet ppSet = PrincipalPosSet.CollectNodesAboveVisitThreshold(rootPos, rootNode, minVisits);
    List<PrincipalPos> principalPositions = ppSet.Members;
    if (verbose)
    {
      Console.WriteLine($"{principalPositions.Count} principal positions after search with root N {rootNode.N}");
    }

    double sumFinal = 0;
    foreach (PrincipalPos principalPosition in principalPositions)
    {
      MGPosition position = principalPosition.LeafPosition;

      if (!position.CalcTerminalStatus().IsTerminal())
      {
        (int numPly, char source, double resultQ, MGPosition finalPos) = RunTerminalPlayout(evaluator, null,
                                                                       new PositionWithHistory(position), false);
        float sideMultiplier = (finalPos.SideToMove == rootPos.SideToMove ? 1 : -1);
        double adjustedQ = sideMultiplier * resultQ;
        sumFinal += adjustedQ;
        string str = adjustedQ < 0 ? "- " : (adjustedQ == 0 ? "  " : "+ ");
        if (verbose)
        {
          Console.WriteLine(str + "      " + numPly + " " + source + " " + position.ToPosition.FEN);
        }
      }
    }

    double avgFinal = sumFinal / principalPositions.Count;

    return avgFinal;
  }



  public static void TestTerminalPlayout(NNEvaluator evaluator)
  {
    evaluator = evaluator ?? NNEvaluator.FromSpecification("~T81", "GPU:0");
    ISyzygyEvaluatorEngine syzygy = SyzygyEvaluatorPool.GetSessionForPaths(CeresUserSettingsManager.Settings.SyzygyPath);

    while (true)
    {
      string fen = Console.ReadLine();
      PositionWithHistory pwh = PositionWithHistory.FromFENAndMovesUCI(fen);
      (int numPly, char source, double resultQ, MGPosition finalPos) result = RunTerminalPlayout(evaluator, syzygy, pwh);
      Console.WriteLine(result);
      Console.WriteLine();
    }
    syzygy.Dispose();
  }

  /// <summary>
  /// Performs a terminal playout - consecutively performs best move in each encountered position.
  /// </summary>
  /// <param name="evaluator"></param>
  /// <param name="syzygy"></param>
  /// <param name="position"></param>
  /// <returns></returns>
  public static (int numPly, char source, double resultQ, MGPosition finalPos) RunTerminalPlayout(NNEvaluator evaluator, 
                                                                             ISyzygyEvaluatorEngine syzygy, 
                                                                             PositionWithHistory position, 
                                                                             bool showDetail = true)
  {
    MGPosition pos = position.FinalPosMG;
    EncodedPositionBatchBuilder batchBuilder = new(1, NNEvaluator.InputTypes.All);
    int numPly = 0;

    HashSet<ulong> posHashes = [];

    float lastV = 0;
    while (true)
    {
      PosHash64WithMove50AndReps hash = MGPositionHashing.Hash64WithMove50AndRepsAdded(in pos, default, default);
      if (posHashes.Contains(hash.Hash))
      {
        return (numPly, 'R', 0, pos); // repetition
      }
      posHashes.Add(hash.Hash);

      batchBuilder.ResetBatch();
      batchBuilder.Add(pos.ToPosition);
      EncodedPositionBatchFlat batch = batchBuilder.GetBatch();

      IPositionEvaluationBatch result = evaluator.EvaluateIntoBuffers(batch);
      numPly++;

      (Memory<CompressedPolicyVector> policies, int policyIndex) = result.GetPolicy(0);
      GameResult terminalStatus = pos.CalcTerminalStatus();
      float v = result.GetV(0);
      MGMove bestMove = terminalStatus.IsTerminal() ? default :  policies.Span[policyIndex].TopMove(pos.ToPosition);

      if (showDetail)
      {
        Console.WriteLine($"  {numPly}  {v} {bestMove} {pos.ToPosition.FEN}");
      }
      
      MGPosition posPrior = pos;

      float resultQ;
      if (terminalStatus.IsTerminal())
      {
        resultQ = terminalStatus == GameResult.Draw ? 0 : -1;
        return (numPly, 'T', resultQ, posPrior);
      }

      pos.MakeMove(bestMove);

      if (pos.Rule50Count >= 99)
      {
        return (numPly, '5', 0, posPrior);
      }
      else if (pos.CheckDrawBasedOnMaterial != Chess.Position.PositionDrawStatus.NotDraw)
      {
        return (numPly, 'M', 0, posPrior);
      }
      else if (syzygy != null && syzygy.ProbeWDLAsV(pos.ToPosition, true) != -999)
      {
        return (numPly, 'S', syzygy.ProbeWDLAsV(pos.ToPosition, true), posPrior);
      }
      else if (MathF.Abs(v) > 0.95f
            && MathF.Abs(lastV) > 0.95f
            && MathF.Sign(v) == MathF.Sign(lastV)) 
      {
        // Adjudicate by NN evaluation (two extreme evals in a row)
        return (numPly, 'V', v > 0.95 ? 1 : -1, posPrior);
      }

      lastV = v;
    }

    return default;
  }


  static void BuildComboNet()
  {
    const string PATH = @"e:\cout\nets\";
    const string BASE_GATE = PATH + "HOP_SP_256_10_8H_FFN4_SMOL_GATE_B1_MUON_4BN_fp16_";
    const string BASE_noGATE = PATH + "HOP_SP_256_12_8H_FFN4_SMOL_noGATE_B1_MUON_4BN_fp16_";

    const string TARGET = PATH + "mix384_80_6late.onnx";
    System.IO.File.Delete(TARGET);
    //const string NF = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_10bn_fp16_";
    //      const string NF = "HOP_SP_768_23_24H_FFN3_SMOL_RPE_GL_B1_MUON_10bn_fp16_";
    //const string NF = "HOP_SP_256_12_8H_FFN4_SMOL_noGATE_B1_MUON_4BN_4kBS_fp16_";
    const string NF = null;// "HOP_SP_256_12_8H_FFN4_SMOL_noGATE_B1_MUON_4BN_fp16_";
    ONNXFileAveraging.CreateAveragedFile(TARGET, [
      //        PATH + NF + "5999984640.onnx",
      //PATH +"HOP_SP_512_55_16H_FFN4_NLA_SMOL_SP_B1_9bn_fp16_7199981568.onnx",
      //PATH + "HOP_SP_512_55_16H_FFN4_NLA_SMOL_SP_B1_9bn_fp16_7999979520.onnx",
      //PATH + "C1-512-55-pre2.onnx",
      //        PATH + "C1-512-55-pre2.onnx",
      //         @"e:\cout\nets\post81bn_11x_nc.onnx"

//     PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_3BN_fp16_2099994624.onnx",
//    PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_2199994368.onnx",

//          PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_2999992320.onnx",
//          PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3099992064.onnx",
//          PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3199991808.onnx",
//       PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3299991552.onnx",
//         PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3399991296.onnx",
//       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3499991040.onnx",
//       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3599990784.onnx",
//                 PATH + "lastc6.onnx",
//          PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3799990272.onnx",
//          PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3899990016.onnx",
//         PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3999989760.onnx",
#if NOT
          PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4099989504.onnx",
          PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4199989248.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4299988992.onnx",
       PATH+ "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4399988736.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4499988480.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4599988224.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4699987968.onnx",
#endif
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4799987712.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4899987456.onnx",
       PATH + "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_5000001024.onnx",
       PATH + "lastf1.onnx",
       PATH + "lastf2.onnx",
       PATH + "laste9.onnx",

 //         PATH+"laste7.onnx"
          
//      PATH + "lastx.onnx",

        ]);

#if NOT
      ONNXFileAveraging.CreateAveragedFile(PATH+"combo_gate.onnx", [
                                                                     BASE_GATE + "3299991552.onnx",
                                                                     BASE_GATE + "3099992064.onnx",
                                                                     BASE_GATE + "2899992576.onnx"]);

      ONNXFileAveraging.CreateAveragedFile(PATH + "combo_nogate.onnx", [ 
                                                                      BASE_noGATE + "3699990528.onnx",
                                                                      BASE_noGATE + "3799990272.onnx",
                                                                      BASE_noGATE + "3899990016.onnx"]);
#endif
  }


  public static void RunTournamentTest()
  {
    //BuildComboNet();System.Environment.Exit(3);//

    // T75 @3k graph vs not:  13 Elo +/-12
    // T75 @30k graph vs not: 44 Elo +/-16
    const int NUM_SEARCH_NODES_1 = 50;
    const int NUM_SEARCH_NODES_2 = 50;
    const bool TEST_VALUE_HEAD = false;

    //string TEST_NET2 = "HOP_SP_256_12_8H_FFN4_SMOL_noGATE_B1_MUON_4BN_fp16_3499991040.onnx";// "~T79"; //"~T75"; ;//"~T1_256_RL_SPSA_TRT"; //"C1-512-35";// "~T3_DISTILL_512_15_FP16_TRT";
    //    string TEST_NET2 = "HOP_CL_CLEAN_256_10_FFN6_B1_NLATT_4bn_fp16_3599990784.onnx";
    //string TEST_NET1 = "C1-512-55-pre1.onnx,combo_768_23_GL_j.onnx";

    // POLICY vs VALUE
    //string TEST_NET1 = "C1-512-55-pre2.onnx;0;1;0;0;0;0,~BT4_FP16_TRT;1;0;1;1;1;1#TensorRT16";
    //string TEST_NET1 = "C1-512-55-pre2.onnx|V2FRAC=0.75;V1TEMP=0.8;V2TEMP=0.8";
    //string TEST_NET2 = "combo_55a";
    //string TEST_NET2 = "C1-640-34";//
    //string TEST_NET1 = "combo_55c"; // slightly better  than pre2 (?)
    //string TEST_NET2 = "C1-512-55-pre2";
    //string TEST_NET1 = "C1-512-15";

    //string TEST_NET1 = "~T1_256_RL_TRT|V1TEMP=0.4";
    //string TEST_NET2 = "~T1_256_RL_TRT";
    //string TEST_NET2 = "~T1_256_RL_NATIVE";

    string TEST_NET1 = "~T3_DISTILL_512_15_FP16_TRT";
    string TEST_NET2 = "~T3_DISTILL_512_15_NATIVE";

    TEST_NET1 = TEST_NET2 = "HOP_SP_C2_384_12_24_attn2x_12H_FFN4_SMOL_NLAeven_MUON_150mm_TEST_fp16_102400.onnx";
    TEST_NET1 = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_10bn_fp16_10000001024.onnx";
    TEST_NET2 = "avg_combo_768_26_10bn_12";

    // CLAIMED BETTER: 999 net
    //TEST_NET1 = "f2c88aecad90_SP_640_34_20H_FFN3_NLA_SMOL_SP_B1_10bn_fp16_9999977472";
    //    TEST_NET1 = "avg_combo_768_26_8claude";

//    TEST_NET1 = "avg_combo_512_55_3claude|V2FRAC=1"; // Wins Value HEad vs 640x34
//    TEST_NET1 = "avg_combo_768_26_8claude|V2FRAC=0.40;V1TEMP=0.95;V2TEMP=1.7;V1_UNC_SCALE=4;V2_UNC_SCALE=4";
//    TEST_NET2 = "avg_combo_768_26_8claude|V2FRAC=0.40;V1TEMP=1.15;V2TEMP=1.9";

    //TEST_NET1 = "avg_combo_768_26_8claude|cudagraphs=true"; // <----------------------
//    TEST_NET1 = @"avg_combo_768_26_8claude_nc-I8.onnx|cudagraphs=true";//avg_combo_768_26_8claude_nc_fixed-I8.onnx
//    TEST_NET1 = "C1-768-26-I8,C1-640-34-I8|V1TEMP=0.35;V2TEMP=1.2;cudagraphs=false";
//    TEST_NET2 = "~BT4_FP16_TRT|cudagraps=true";

//   TEST_NET1 = "avg_combo_768_26_8claude_nc-I8|cudagraphs=true";
    TEST_NET1 = "C1-768-26-I8|cudagraphs=true;bf16=true;V2FRAC=0.0;V1TEMP=1.32";
    TEST_NET2 = "C1-640-34-I8|cudagraphs=true;bf16=true";

    //TEST_NET1 = "C1-512-15-I8|V1TEMP=0.45;V2TEMP=1.4;V1_UNC_SCALE=2;V2_UNC_SCALE=2";
    // -20
    //TEST_NET1 = "C1-512-15-I8|cudagraphs=true;bf16=true;V2FRAC=0;V1TEMP=0.65;V1_UNC_SCALE=-0.001";    
    //TEST_NET2 = "C1-512-15-I8|cudagraphs=true;bf16=true;V2FRAC=0";

//    TEST_NET1 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_d512x25_6bn_fp16_last.onnx";
//    TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_6bn_fp16_299999232.onnx";
//     TEST_NET2 = "C2-384-12-beta1-I8.onnx|bf16=true;cudagraphs=true";

//TEST_NET1 = "C1-512-55-pre2-I8|cudagraphs=true;V2FRAC=0.0;V1TEMP=0.66;bf16=true";
TEST_NET1 = "C1-512-55-pre2|cudagraphs=true;bf16=true;V2FRAC=0.375;V1TEMP=0.95;V2TEMP=1.85";
TEST_NET2 = "C1-640-34-I8|cudagraphs=true;bf16=true";
// DISTILL TEST   
TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_d512x25_6bn_fp16_1199996928|bf16=true;cudagraphs=true";
TEST_NET1 = "avg_combo_384d_12bn.onnx|bf16=true;cudagraphs=true";

    //TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_d512x25_6bn_fp16_899997696|bf16=true;cudagraphs=true";
    //TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_6bn_fp16_1799995392|bf16=true;cudagraphs=true";

    //    TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_d512x25_6bn_fp16_899997696|V2FRAC=1;bf16=true;cudagraphs=true";
    //    TEST_NET2 = "C1-640-34-I8";

    //TEST_NET1 = "avg_combo_C2_384_claude1"; // <--------- possibly best according to Claude
    TEST_NET2 = "C2-384-12-beta1-I8.onnx|bf16=true;cudagraphs=true";

    TEST_NET1 = "C1-256-10-claudeopt|cudagraphs=true;bf16=true";
    TEST_NET2 = "C1-256-10-I8|cudagraphs=true;bf16=true";

    //TEST_NET1 = "avg_combo_512_55_5claude_nc|cudagraphs=true;bf16=true";
    TEST_NET1 = "C1-640-34,avg_HOP_SP_512_55_16H_FFN4_NLA_SMOL_SP_B1_9bn_fp16_4nets_valopt|cudagraphs=false;bf16=true";
    TEST_NET2 = "C1-640-34-I8|cudagraphs=true;bf16=true";

    TEST_NET1 = "avg_combo_384d_27bn4|cudagraphs=true;bf16=true";
  //  TEST_NET1 = "avg_combo_384d_24bn3|cudagraphs=true;bf16=true";// ;v2frac=0";
    TEST_NET2 = "C2-384-12-beta1-I8.onnx|bf16=true;cudagraphs=true";// ;v2frac=0";

    TEST_NET1 = "avg_combo_384d_30bn3|cudagraphs=true;bf16=true";
    //TEST_NET2 = "avg_combo_384d_21bn4|cudagraphs=false";

    TEST_NET1 = "C1-640-34-I8|bf16=false;cudagraphs=true";
    TEST_NET2 = "C1-640-34-I8|bf16=true;cudagraphs=true";

    //TEST_NET2 = "HOP_SP_C2_384_12_12H_FFN4_SMOL_NLA_MUON_d512x25_6bn_fp16_1499996160.onnx";
    //TEST_NET1 = TEST_NET2 = "C1-384-12-i8|cudagraphs=true;bf16=true";

    //TEST_NET1 = "C1_640_34_10bn_rewrite|cudagraphs=true;bf16=true";
    //TEST_NET2 = "C1-640-34-I8|cudagraphs=true;bf16=true";

    //    f2c88aecad90_SP_640_34_20H_FFN3_NLA_SMOL_SP_B1_10bn_fp16_9999977472
#if NOT
    TEST_NET1 = "HOP_SP_512_55_16H_FFN3_NLA_SMOL_SP_B1_8bn_BASE_fp16_7299981312|cudagraphs=true;bf16=true";
    TEST_NET2 = "HOP_SP_512_55_16H_FFN4_NLA_SMOL_SP_B1_9bn_fp16_7299981312|cudagraphs=true;bf16=true";   
    TEST_NET2 = "C1-512-55-pre2-I8|cudagraphs=true;bf16=true";

// promising?    TEST_NET1 = "avg_HOP_SP_512_55_16H_FFN4_NLA_SMOL_SP_B1_9bn_fp16_4nets_valopt|bf16=true;V2FRAC=0.41;V1TEMP=0.93;V2TEMP=1.83";
    TEST_NET1 = "avg_combo_512_55_2claude_nc|bf16=true;V2FRAC=0.36;V1TEMP=0.95;V2TEMP=1.8";
    TEST_NET2 = "C1-512-55-pre2-I8|bf16=true;V2FRAC=0.36;V1TEMP=0.95;V2TEMP=1.8";
#endif
    //string TEST_NET1 = "avg_combo_640_28_5l_nc.onnx";
    //string TEST_NET2 = "C1-640-34-i8";

    //string TEST_NET2 = "~T1_DISTILL_256_10_FP16_NATIVE";
    //    string TEST_NET1 = "~T81";
    //TEST_NET2 = TEST_NET1;
    //  string TEST_NET2 = "~T1_DISTILL_256_10_NATIVE";
    //    TEST_NET2 = "~T74";
    //    string TEST_NET1 = "C1-384-12-i8|cudagraphs=true";
    //    string TEST_NET2 = TEST_NET1;
    //   string TEST_NET1 = "~T3_DISTILL_512_15_FP16_TRT|cudagraphs=true";

    //    TEST_NET2 = "HOP_SP_384_12_12H_FFN3_B1_MUON_750MM_fp16_199999488.onnx|cudagraphs=true"; // <<- baseline
    //    TEST_NET2 = "HOP_SP_384_12_dyn_20b_int512_NLA_8H_FFN3_MUON_5bn_fp16_199999488.onnx|cudagraphs=true";// longrun


    //    TEST_NET2 = "d4ae6e742b90_SP_640_35_20H_FFN3_NLA_SMOL_SP_B1_9bn_fp16_600001536.onnx";
    //string TEST_NET1 = "avg_combo_640_30_4l_nc.onnx|cudagraphs=false";
    //    string TEST_NET1 = "HOP_SP_C2_640_30_20H_FFN3_SMOL_NLA_MUON_8bn_fp16_3599990784";
    //    string TEST_NET2 = "C1-640-34-i8|cudagraphs=true";

    //    string TEST_NET1 = "C1-512-55-pre2-I8|V2FRAC=0.25;V1TEMP=0.60"; // add I8

    // 0.5/1.3 --> 10 +/-15
    // 0.5/1.0 --> 20 +/-15
    // 0.75/1.0 --> 19 +/- 15
    // 0.5/0.85 --> 14 +/- 16

    // 0.6/1.1 @500 --> 6 +/- 11
    // 0.85/1.1 @500 -->-15 +/- 20 
    // 0.4/1.1 @500 --> 4 +/- 15
    // 0.5/0.9 @500 --> -15 +/- 17
    // 0.6/1.2 @2500 --> 2 +/- 13
    //    string TEST_NET1 = "C1-512-55-pre1.value3_L53_ReferenceEngine_x1_d2.onnx.value3|V2FRAC=0.6;V2TEMP=1.2";

    //    string TEST_NET1 = @"e:\cout\nets\avg_combo_640_57_5l_nc.onnx|cudagraphs=true";// ;V2FRAC=1;V1TEMP=1;V2TEMP=1";
    //    string TEST_NET1 = "avg_combo_640_75_5_nc.onnx|V1TEMP=0.85";// ;V2FRAC=1;V1TEMP=1;V2TEMP=1";
    //    string TEST_NET2 = "C1-640-34-i8|cudagraphs=true";// ;V2FRAC=1;V1TEMP=1;V2TEMP=1";

    //    TEST_NET1 = TEST_NET2 = "~BT4_FP16_TRT|cudagraphs=true";
    //TEST_NET1 = TEST_NET2 = "~T1_DISTILL_512_15_FP16|cudagraphs=true";
    //    TEST_NET1 = "~T1_DISTILL_256_10_FP16_TRT|cudagraphs=true";
    //    TEST_NET2 = "~T1_DISTILL_256_10_NATIVE|cudagraphs=true";
    //TEST_NET2 = TEST_NET1;

    //TEST_NET1 = TEST_NET2 = "C2-384-12-beta1-I8.onnx|cudagraphs=true";

    //  TEST_NET1 = "avg_combo_640_72_4l_nc|cudagraphs=true";
    ///    TEST_NET2 = "9ae67f7c4630_SP_640_34_20H_FFN3_NLA_SMOL_SP_B1_10bn_fp16_4999990272.onnx";
    //    TEST_NET1 = TEST_NET2 = "C1-640-34-I8.onnx|cudagraphs=true;V2FRAC=0;V1TEMP=1";

    //TEST_NET1 = TEST_NET2 = "C1-640-34-i8|cudagraphs=true";

    //    TEST_NET1 = TEST_NET2 = "C2-384-12-beta1-I8.onnx|cudagraphs=true";

    //    TEST_NET1 = "avg_combo_384_57_2_l_nc-I8.onnx|cudagraphs=true"; // very good <---------
    //    TEST_NET2 = "C1-384-12-I8|cudagraphs=true";

    //    TEST_NET2 = "C2-384-12-beta1-I8.onnx|cudagraphs=true";

    //    TEST_NET1 = "HOP_SP_C2_640_30_20H_FFN3_SMOL_NLA_MUON_8bn_fp16_2199994368.onnx";
    //    TEST_NET2 =    "d4ae6e742b90_SP_640_35_20H_FFN3_NLA_SMOL_SP_B1_9bn_fp16_2249994240.onnx";

    //    TEST_NET2 = "C1-384-12-i8|cudagraphs=true";

    //    TEST_NET1 = "HOP_SP_C2_640_30_20H_FFN3_SMOL_NLA_MUON_8bn_fp16_1999994880.onnx|cudagraphs=true";
    //    TEST_NET2 = "d4ae6e742b90_SP_640_35_20H_FFN3_NLA_SMOL_SP_B1_9bn_fp16_2024994816.onnx|cudagraphs=true";

    //TEST_NET2 = "C1-256-10-i8|cudagraphs=true";

    //  TEST_NET1 ="C1-256-10-i8|cudagraphs=true";
    //TEST_NET1 = TEST_NET2 = "~T81";

    //    TEST_NET1 = "mix384_80_6late|V1TEMP=0.55;cudagraphs=false";
    //    TEST_NET2 = "C1-640-34-i8|cudagraphs=false";

    //TEST_NET1 = TEST_NET2 = "~T79";

    //TEST_NET1 = "HOP_SP_C2_640_30_20H_FFN3_SMOL_NLA_MUON_8bn_fp16_last.onnx|V2FRAC=1;cudagraphs=true";
    //TEST_NET2 = "HOP_SP_C2_640_30_20H_FFN3_SMOL_NLA_MUON_8bn_fp16_2799992832.onnx|V2FRAC=1;cudagraphs=true";

    //    string TEST_NET2 = "~BT4_FP16_TRT";
    //TEST_NET2 = "~BT4_NATIVE";
    //    TEST_NET1 = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_10bn_fp16_7099981824.onnx";

    //string TEST_NET1 = "~T74_SPSA";// "C1-512-15-I8";

    //TEST_NET1 = TEST_NET2 = "~T81";
    //TEST_NET1 = TEST_NET2 = "badgyal-3.pb.gz";// "~T74";

    //TEST_NET1 = "ONNX_TRT:744706_spsa032025.pb_fp16.onnx";
    //TEST_NET2 = "~T74_SPSA";

    //    TEST_NET1 = "mix384_s";
    //    TEST_NET2 = "C1-512-25";

    //    string TEST_NET1 = "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3499991040.onnx";
    //        TEST_NET1 = "last7.onnx";
    //    TEST_NET2 = "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_2199994368";

    //   string TEST_NET2 = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_7bn_fp16_3499991040";
    //    TEST_NET1 = TEST_NET2 = "C1-512-55-pre2";


    //    TEST_NET1 = "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3699990528";
    //    TEST_NET2 = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_7bn_fp16_3699990528";
    //    TEST_NET1 = "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_4399988736";

    //    TEST_NET1 = "C1-640-34-i8|cudagraphs=true";
    //    TEST_NET2 = "C1-640-34-i8|cudagraphs=true";
    //    TEST_NET1 = "mix384_3_45bn";
    //TEST_NET1 = "mix384_6_49bn";// "mix384_4_485bn";
    //    TEST_NET2 = "HOP_SP_384_80_8H_FFN2_GL4_SMOL_B1_MUON_5BN_fp16_3999989760";
    //TEST_NET2 = "~BT4_FP16_TRT";
    //TEST_NET2 = "C1-640-34-i8";
    //    TEST_NET2 = "9ae67f7c4630_SP_640_34_20H_FFN3_NLA_SMOL_SP_B1_10bn_fp16_4399991808_nc";
    //    TEST_NET2 = "C1-640-34-i8";

    //    TEST_NET1 = TEST_NET2= "badgyal-3.pb.gz";

    // Test ONNXExecutor
    //    TEST_NET1 = "C1-640-34-I8";
    //    TEST_NET2 = "C1-640-34-i8";

    if (false)
    {
//      TEST_NET1 = "C1-640-34-i8";//|V2FRAC=0.4;V1TEMP=0.4;V2TEMP=1.2";
//      TEST_NET2 = "~BT4_FP16_TRT";
    }

    //TEST_NET1 = "~BT4_FP16_TRT|cudagraphs=true";

//    TEST_NET1 = "C1-256-10-i8|cudagraphs=true";
//    TEST_NET2 = "C1-256-10-i8|cudagraphs=false";

    //TEST_NET1 = TEST_NET2 = "~T81_FP16_TRT";

    //TEST_NET1 = "ONNX_TRT:BT4-1024x15x32h-swa-6147500-copy.pb.gz_fp16";
    //TEST_NET2 = "ONNX_TRT:BT4-1024x15x32h-swa-6147500.pb.gz_fp16";

    //    TEST_NET2 = "C1-640-34-i8";
    //TEST_NET1 = "~BT4_FP16_TRT";
    //TEST_NET1 = "C1-256-10-i8";
    //TEST_NET1 = TEST_NET2 = "~T81";

    //    TEST_NET1 = "mix384_80_4late";

    //    TEST_NET1 = TEST_NET2 = "~BT2_FP16_TRT";
    //TEST_NET1 = TEST_NET2 = "~T81";

    //    TEST_NET1 = "ONNX_TRT:t1-256x10-distilled-swa-2432500_fp16.onnx";
    //TEST_NET2 = TEST_NET1;
    //    string TEST_NET2 = "C1-640-34-i8|V2FRAC=0.0;V1TEMP=1.000;V2TEMP=1.5";

    //C:\apps\lc0_32\lc0 leela2onnx --input=d:\nets\weights_run3_900000.pb.gz --onnx-data-type=f16  --output=d:\nets\weights_run3_900000.pb.gz.onnx
    //    TEST_NET1 = @"ONNX_TRT:d:\nets\weights_run3_900001.pb.gz.onnx";
    //TEST_NET2 ="~BT4_FP16_TRT";

    //  TEST_NET1 = "mix384_u";
    //string TEST_NET2 = "C1-512-25";
    //TEST_NET2 = "HOP_SP_768_26_24H_FFN3_SMOL_RPE_B1_MUON_7bn_fp16_499998720";

    //TEST_NET1 = "~BT2_FP16";
    //TEST_NET2 = "~BT2_NATIVE";

    // NOTE: for matched (approximately) nets use "C1-512-15" with ("~BT2_NATIVE" or "~BT2")
    //NNEvaluatorOptionsCeres optionsCeres1 = new NNEvaluatorOptionsCeres();
    //optionsCeres1.QNegativeBlunders = 0.09f;
    //optionsCeres1.QPositiveBlunders = 0.09f;

    string DEVICE1 = "GPU:0#TensorRTNative";
    string DEVICE2 = "GPU:0#TensorRT16";

    const bool USE_SF = false;
    const bool USE_CERES_V1_MCTS_FOR_ENGINE2 = false;

    bool RUN_SUITE = false;

#if NOT
    // ...............................................
    GameEngineDef gameEngine2 = null;

    const bool USE_UCI2_CERES = false;
    gameEngine2 = USE_UCI2_CERES ? GameEngineCeresUCI(TEST_NET1) : gameEngine2;

    const bool USE_UCI2_LC0 = false;
    const bool USE_UCI2_LC0_NONPREVIEW = false;
    gameEngine2 = USE_UCI2_LC0 ? GameEngineLc0(TEST_NET1, false, USE_UCI2_LC0_NONPREVIEW) : gameEngine2;

    const bool USE_UCI2_LC0_DAG = false;
    gameEngine2 = USE_UCI2_LC0_DAG ? GameEngineLc0(TEST_NET1, true) : gameEngine2;
    // ...............................................

    // ...............................................
const bool USE_UCI1_LC0 = true;
    GameEngineDef gameEngine1 = null;
    if (USE_UCI1_LC0)
    {
      gameEngine1 = GameEngineLc0(TEST_NET1, true);
//        gameEngine2Alternate = GameEngineLc0(TEST_NET1, false, true);
    }
    // ...............................................
//gameEngine1Alternate = GameEngineCeresUCI(TEST_NET1);
#endif

    const int SF_MULTIPLIER = 20;
    SearchLimit searchLimitTournament1 = SearchLimit.NodesPerMove(NUM_SEARCH_NODES_1);
    SearchLimit searchLimitTournament2 = SearchLimit.NodesPerMove(USE_SF ? NUM_SEARCH_NODES_1 * SF_MULTIPLIER : NUM_SEARCH_NODES_2);

    if (TEST_VALUE_HEAD)
    {
      searchLimitTournament1 = SearchLimit.BestValueMove;
      searchLimitTournament2 = SearchLimit.BestValueMove;
    }

    if (RUN_SUITE && !USE_CERES_V1_MCTS_FOR_ENGINE2)
    {
      //throw new Exception("combination NOT supported");
    }

    NNEvaluatorDef evalDefMCGS1 = NNEvaluatorDef.FromSpecification(TEST_NET1, DEVICE1);//, optionsCeres1);//
    NNEvaluatorDef evalDefMCGS2 = NNEvaluatorDef.FromSpecification(TEST_NET2, DEVICE2);

//evalDefMCGS1.Options = new NNEvaluatorOptions() with { UseMiddlegameSlowdown = true}; // DJE

    //evalDefMCGS1.Options = new NNEvaluatorOptionsCeres() with { UseMiddlegameSlowdown = true }; // DJE
    //GraphTestingIntegration.TestRecursivePreloadTree(); System.Environment.Exit(3);

    if (true)
    {
      if (!USE_CERES_V1_MCTS_FOR_ENGINE2)
      {
        Console.WriteLine("MCGS1 vs MCGS2 differences");
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(SEARCH_PARAMS_MCGS, SEARCH_PARAMS_MCGS2, true));
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(SEARCH_PARAMS_MCGS.Execution, SEARCH_PARAMS_MCGS2.Execution, true));
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(SELECT_PARAMS_MCGS, SELECT_PARAMS_MCGS2, true));
        Console.WriteLine();
      }

      GameEngineDef engine1Def = new GameEngineDefCeresMCGS("MCGS1", evalDefMCGS1, SEARCH_PARAMS_MCGS, SELECT_PARAMS_MCGS);
      GameEngineDef engine2Def = new GameEngineDefCeresMCGS("MCGS2", evalDefMCGS2, SEARCH_PARAMS_MCGS2, SELECT_PARAMS_MCGS2);

      const float MULT = 1f;
      const float MULT_EXTRA_2 = USE_SF ? 0.5f : 1;
      //searchLimitTournament1 = new SearchLimit(SearchLimitType.SecondsForAllMoves, 30 * MULT, false, 0.5f * MULT);
      //searchLimitTournament2 = searchLimitTournament1 * MULT_EXTRA_2;
      
//      searchLimitTournament1 = new SearchLimit(SearchLimitType.SecondsForAllMoves, 30*60, false, 3f);
//      searchLimitTournament2 = searchLimitTournament1 * MULT_EXTRA_2;

      const bool RUN_VERSUS = false;
      if (RUN_VERSUS)
      {
        GameEngineDefCeresMCGS engineDefCeresMCTS = new("MCGS", evalDefMCGS1, SEARCH_PARAMS_MCGS, SELECT_PARAMS_MCGS, true);
        throw new NotImplementedException("reenable next line");
        //TournamentTest.RunEngineComparisons(engineDefCeresMCTS.CreateEngine(), "r3r2k/1p4pp/2pB1p2/2P4q/1Q6/P4N2/6PP/R4K2 b - - 0 29");
      }
      if (USE_CERES_V1_MCTS_FOR_ENGINE2)
      {
        GameEngineDefCeres engineDefCeresMCTS = new("MCTS", evalDefMCGS2, null, 
                                                    SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS), 
                                                    new MCTS.Params.ParamsSelect());
        TournamentTest.Test(engine1Def, engineDefCeresMCTS, 
                            overrideSearchLimit1: searchLimitTournament1, 
                            overrideSearchLimit2: searchLimitTournament2);

//          engine2Def = new GameEngineDefCeres("MCTS", evalDefMCGS2, null,
//                                               SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS), 
//                                               new Ceres.MCTS.Params.ParamsSelect());

      }
      else
      {
// ................................          
//engine2Def = GameEngineLc0(TEST_NET2, LC0EngineType.RewriteClassic);
//engine2Def = GameEngineLc0(TEST_NET2, DEVICE2, LC0EngineType.RewriteDAG);

//          engine2Def = GameEngineLc0(TEST_NET2, LC0EngineType.TCEC_DAG);

//        engine2Def = new GameEngineDefCeresMCGS("MCGS2", evalDefMCGS1, SEARCH_PARAMS_MCGS2, SELECT_PARAMS_MCGS2);
// ................................

        RunTournamentTestDirect(evalDefMCGS1, evalDefMCGS2,
                                engine1Def, engine2Def,
                                searchLimitTournament1, searchLimitTournament2,
                                RUN_SUITE,
                                USE_SF);
        }
      DumpNNStats();
      System.Environment.Exit(3);
    }


    // ============
    for (int i = 0; i < 1; i++)
    {
      //        TestStateUsefulness.TestImpactOfStateEnabledOnValueAccuracy();
      //        GraphTestingIntegration.GraphIntegrationTest();
    }

    System.Environment.Exit(3);


    //      NNEvaluator evaluator = NNEvaluator.FromSpecification("703810", "GPU:0");
    string fen = "8/2p2k1p/ppPp1p2/5Pp1/2P1K1P1/7P/PP6/8 w - - 0 1"; // #665 Averbakh
    fen = Position.StartPosition.FEN;
    //      fen = "8/3k4/8/8/8/4K3/2Q5/8 b - - 0 1"; // easy checkmate soon

    NNEvaluatorDef evalDef = NNEvaluatorDef.FromSpecification(TEST_NET1, "GPU:0");
    MCTS.Params.ParamsSearch paramsSearch = new();
    MCTS.Params.ParamsSelect paramsSelect = new();
    //      paramsSearch.Execution.MaxBatchSize = 1;
    paramsSearch.Execution.SelectParallelEnabled = false;

    GameEngineCeresInProcess engine = new("CeresMCGS", evalDef, null, paramsSearch, new MCTS.Params.ParamsSelect());

    engine.SearchParams.Execution.TranspositionMode = MCTS.Params.TranspositionMode.None;

    const int SEARCH_LENGTH = 20;// 8_000_000;// 1000;
    SearchLimit searchLimit = SearchLimit.NodesPerMove(SEARCH_LENGTH);// 0_000_000);

    GameEngineSearchResultCeres engineResult = null;
    engineResult = engine.SearchCeres(PositionWithHistory.FromFENAndMovesUCI(fen, ""), SearchLimit.NodesForAllMoves(1));
    engineResult.Search.Manager.Context.Tree.Store.Validate(engineResult.Search.Manager.Context.Tree.TranspositionRoots);

    for (int i = 0; i < 1; i++)
    {
      engine.ResetGame();
      using (new TimingBlock("tb"))
      {
        engineResult = engine.SearchCeres(PositionWithHistory.FromFENAndMovesUCI(fen, ""), searchLimit);

        if (engineResult.FinalN < 100)
        {
          engineResult.Search.Manager.Context.Tree.Store.Dump(true);
          //            System.Environment.Exit(3);
        }

      }
    }

    Console.WriteLine(engineResult);
    //      engineResult.Search.Manager.Context.Tree.Store.Dump(false);
    engineResult.Search.Manager.DumpFullInfo(Console.Out, fen);
    return;
  }


  private static GameEngineDef GameEngineCeresUCI(string TEST_NET1)
  {
    static string exeCeres() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net8.0/Ceres.dll"
                                    : @"C:\dev\ceres\artifacts\release\net8.0\ceres.exe";
    NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(TEST_NET1, "GPU:0");
    GameEngineDefCeresUCI engineDefCeresUCI = new("CeresUCINew", evalDef1, overrideEXE: exeCeres(),
                                                  disableFutilityStopSearch: !SEARCH_PARAMS_MCGS_COMMON.FutilityPruningStopSearchEnabled,
                                                  paramsSearch: SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS),
                                                  paramsSelect: new Ceres.MCTS.Params.ParamsSelect());
    return engineDefCeresUCI;
  }

  public enum LC0EngineType
  {
    Legacy,
    RewriteClassic,
    RewriteDAG,
    RewriteDAG_CUDA,
    RewriteDAGPrior,
    TCEC_DAG
  }

  public static GameEngineDef GameEngineLc0(string TEST_NET, string DEVICE, LC0EngineType type, 
                                            bool disableFutilityPruning = false, bool verboseMoveOutput = false,
                                            string extraUCIOptions = null)
  {
    (string EXE, string extraArgs) lc0Info = type switch
    {
      LC0EngineType.Legacy => (null, ""),
      LC0EngineType.RewriteClassic => 
        SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "")
//        : (@"C:\apps\lc0_32\lc0.exe", ""),
//        : (@"C:\apps\lc0_32\lc0_33pre-trt.exe", ""),
          : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", ""),
      //        : (@"c:\apps\lc0_32\non-onnx\lc0.exe", ""),


#if EXE_VALIDATION_METRIC

1. Linux CUDA
~/dev/lc0_menkib/lc0/build/release$ ./lc0-cuda --backend=cuda-fp16 -w /mnt/devd/nets/791556.pb.gz
info depth 20 seldepth 64 time 62226 nodes 9658711 score cp 27 hashfull 1000 nps 161733
./lc0-cuda backendbench --backend=cuda-fp16 -w /mnt/devd/nets/791556.pb.gz --start-batch-size=8 --batch-step=8
Benchmark batch size 256 with inference average time 1.63427ms - throughput 156645 nps.

2. Windows CUDA
C:\apps\lc0_swiss9\lc0\build>lc0-cuda dag-preview --backend=cuda-fp16 -w d:\nets\791556.pb.gz
info depth 20 seldepth 64 time 54181 nodes 9807098 score cp 28 hashfull 1000 nps 182081
lc0-cuda backendbench --backend=cuda-fp16 -w d:\nets\791556.pb.gz --start-batch-size=8 --batch-step=8
Benchmark batch size 256 with inference average time 2.49875ms - throughput 102451 nps.


3. Linux ONNX
david@HOP:/home/david/dev/lc0_321/lc0/build/release/lc0 dag-preview --backend=onnx-trt -w /mnt/devd/nets/t1-256x10-distilled-swa-2432500.pb.gz
info depth 17 seldepth 61 time 60780 nodes 4461875 score cp 16 nps 124261
./lc0 backendbench --backend=onnx-trt -w /mnt/devd/nets/t1-256x10-distilled-swa-2432500.pb.gz --start-batch-size=8 --batch-step=8
256,   112360,   2.278, 0.007,  112979,  112398,  110153

3. Windows ONNX
C:\apps\lc0_32>lc0_lepned_eps_1Nov.exe dag-preview  --backend=onnx-trt -w d:\nets\t1-256x10-distilled-swa-2432500.pb.gz
info depth 17 seldepth 60 time 61610 nodes 4912039 score cp 16 nps 81481
lc0_lepned_eps_1Nov.exe backendbench --start-batch-size=8 --batch-step=8 -w d:\nets\t1-256x10-distilled-swa-2432500.pb.gz --backend=onnx-trt
Benchmark batch size 256 with inference average time 4.50483ms - throughput 56827.9 nps.

#endif
      LC0EngineType.RewriteDAG => 
        SoftwareManager.IsLinux ? ("/home/david/dev/lc0_321/lc0/build/release/lc0", "dag-preview")
                                : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", "dag-preview"),

      //  : (@"C:\apps\lc0_32\lc0-321-trt.exe", "dag-preview"), OFFICIAL 0.32.1 but slow DAG!

      LC0EngineType.RewriteDAG_CUDA 
        => SoftwareManager.IsLinux ? ("/home/david/dev/lc0_menkib/lc0/build/release/lc0-cuda", "dag-preview")
                                   : (@"C:\apps\lc0_swiss9\lc0\build\lc0-cuda.exe", "dag-preview"),


      LC0EngineType.RewriteDAGPrior => SoftwareManager.IsLinux ? ("/home/david/dev/lc0/build/release/lc0", "dag-preview")
                                                              : (@"C:\apps\lc0_32\lc0_lepned_eps_1Nov.exe", "dag-preview"),
                                                              // : (@"C:\apps\lc0_32\lc0.exe", "dag-preview"),      
                                                                //: (@"c:\apps\lc0_32\non-onnx\lc0_dag-preview.exe", ""),

      //c:\apps\lc0_32\non-onnx\lc0_dag-preview.exe
      LC0EngineType.TCEC_DAG => SoftwareManager.IsLinux ? ("/home/david/dev/LC0_DAG_TCEC/lc0/build/release/lc0", "")
                                                        : throw new NotImplementedException(),
      _ => throw new NotImplementedException()
    };

    
    NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(TEST_NET, DEVICE);
    GameEngineDefLC0 engineDefLC1 = new(type.ToString(), evalDef1,
                                        disableFutilityPruning || !SEARCH_PARAMS_MCGS_COMMON.FutilityPruningStopSearchEnabled,
                                        null, null,
                                        overrideEXE: lc0Info.EXE,
                                        alwaysFillInHistory: false,
                                        verbose: verboseMoveOutput,
                                        extraCommandLineArgs: lc0Info.extraArgs + " " + extraUCIOptions,
                                        disableTreeReuse: !SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS).TreeReuseEnabled);
    return engineDefLC1;
  }


  private static void ResetNNStats()
  {
    MCGSEngine.TotalNumNodesPrefetched = 0;
    SelectTerminatorPrefetched.NumNodesSelectedFromPrefetch = 0;
    GraphPrefetcher.TotalNumRearranged = 0;
    GraphPrefetcher.TotalNumPrefetched = 0;
    MCGSEvaluatorNeuralNet.TOTAL_NUM_NN_EVALS = 0;
    LeafEvaluatorNN.TOTAL_NUM_NN_EVALS = 0;
  }

  private static void DumpNNStats()
  {
    Console.WriteLine();
    Console.WriteLine("Num prefetched         : " + MCGSEngine.TotalNumNodesPrefetched);
    Console.WriteLine("Num used from prefetch : " + SelectTerminatorPrefetched.NumNodesSelectedFromPrefetch);
    Console.WriteLine("Num rearranged         : " + GraphPrefetcher.TotalNumRearranged);
    Console.WriteLine();
    Console.WriteLine("Num NN evals (MCGS)    : " + MCGSEvaluatorNeuralNet.TOTAL_NUM_NN_EVALS);
    Console.WriteLine("Num NN evals (MCTS)    : " + LeafEvaluatorNN.TOTAL_NUM_NN_EVALS);
    Console.WriteLine();
    Console.WriteLine("Num NN evals ratio     : " + MathF.Round((float)MCGSEvaluatorNeuralNet.TOTAL_NUM_NN_EVALS / (float)LeafEvaluatorNN.TOTAL_NUM_NN_EVALS, 2));
    Console.WriteLine();
  }


  private static void RunTournamentTestDirect(NNEvaluatorDef evalDef1, NNEvaluatorDef evalDef2,
                                              GameEngineDef engine1Def, GameEngineDef engine2Def,
                                              SearchLimit searchLimit1, SearchLimit searchLimit2, 
                                              bool runSuite = false, bool useSFForEngine2 = false)
  {
    if (runSuite)
    {
      // TODO: remove hardcoded EXE name
      throw new NotImplementedException("BE VERY CAREFUL OF RECURSION - computer crash");
      const string EXE_FN = @"g:\dev\Ceres.MCGS\artifacts\release\net9.0\ceres.mcgs.uci.exe";
      GameEngineDefCeresMCGSUCI engineCeresMCGSExternal = new("CeresMCGS1", evalDef1, overrideEXE: EXE_FN);

      GameEngineDefCeres engineDefCeresMCTS = new("MCTS", evalDef1, evalDef2, SEARCH_PARAMS_MCTS(SEARCH_PARAMS_MCGS), new MCTS.Params.ParamsSelect());
      TournamentTest.Test(engineDefCeresMCTS, null, 
                          overrideNET1:engineDefCeresMCTS.EvaluatorDef.Nets[0].Net.ShortID, 
                          overrideNET2: "NONE", 
                          overrideSearchLimit1: searchLimit1, 
                          overrideSearchLimit2: searchLimit2, 
                          overrideExternalEngineDef: engineCeresMCGSExternal, 
                          runSuite:runSuite);
    }
    else if (useSFForEngine2)
    {
      const int NUM_THREADS = 10; // allows only rare oversubscription  when all 4 matches have SF on the move

      GameEngineDef engineDefSF = MakeEngineDefStockfish("SF17", SF17_1_EXE, numThreads: NUM_THREADS);
      TournamentTest.Test(engine1Def, engineDefSF, overrideSearchLimit1: searchLimit1, overrideSearchLimit2: searchLimit2);
    }
    else
    {
      TournamentTest.Test(engine1Def, engine2Def, overrideSearchLimit1: searchLimit1, overrideSearchLimit2: searchLimit2);
    }
  }


  private static void LaunchUCI(Span<string> args, Action<ParamsSearch> searchModifier, Action<ParamsSelect> selectModifier)
  {
    string cmdArg = args.Length < 2 ? "" : args[1];

    int numToSkip = (args.Length > 1) ? 2 : 1;
    args = args[numToSkip..];
    StringBuilder allArgs = new();
    if (!args.IsEmpty)
    {
      for (int i = 0; i < args.Length; i++)
      {
        allArgs.Append((i != 0 ? " " : "") + args[i]);
      }
    }

    string keyValueArgs = allArgs.ToString();

    if (cmdArg == "backendbench")
    {
      FeatureBenchmarkBackend backendBench = new FeatureBenchmarkBackend();
      backendBench.ParseFields(keyValueArgs);
      backendBench.ExecuteBenchmark(null, null);
      System.Environment.Exit(0);
    }
    else if (cmdArg == "benchmark")
    {
      FeatureBenchmarkSearch benchmarkParams = FeatureBenchmarkSearch.ParseBenchmarkCommand(keyValueArgs);
      benchmarkParams.Execute();
      System.Environment.Exit(0);
    }


    FeatureUCIParams uciParams = FeatureUCIParams.ParseUCICommand(keyValueArgs);

    Action<NNEvaluatorDef, NNEvaluator, int, int, int, int> backendBenchEvaluator = delegate (NNEvaluatorDef evalDef, NNEvaluator evaluator,
                                                                                              int minSize, int maxSize, int stepSize, int repeatCount)
    {
      FeatureBenchmarkBackend backendBench = new();
      (NNEvaluator, List<(int, float)>) speed = backendBench.ExecuteBenchmark(evalDef, evaluator, repeatCount, minSize, maxSize, stepSize);
      Console.WriteLine();
    };

    static void searchBenchmarkAction(NNEvaluatorDef evalDef, int secondsPerMove)
    {
      FeatureBenchmarkSearch.Benchmark(evalDef, SearchLimit.SecondsPerMove(secondsPerMove), false, int.MaxValue);
      Console.WriteLine();
    }

    UCIManagerMCGS ux = new (uciParams.NetworkSpec, uciParams.DeviceSpec,
                                   searchModifier, 
                                   selectModifier, 
                                   null, null, null,
                                   uciParams.Pruning == false,
                                   CeresUserSettingsManager.Settings.UCILogFile,
                                   CeresUserSettingsManager.Settings.SearchLogFile,
                                   backendBenchEvaluator,
                                   searchBenchmarkAction,
                                   SEARCH_PARAMS_MCGS, SELECT_PARAMS_MCGS);

    Console.WriteLine();
    Console.WriteLine("Entering UCI command processing mode.");
    ux.PlayUCI();
  }



  public static void RunEngineComparisons()
  {
    string pgnFileName = SoftwareManager.IsWindows ? @"\\synology\dev\chess\data\pgn\raw\ceres_big.pgn"
                                             : @"/mnt/syndev/chess/data/pgn/raw/ceres_big.pgn";
    string NET_ID = "~T79";

    CompareEngineParams parms = new ("Baseline", 
                                     pgnFileName,
                                     10, // number of positions
                                     s => true,//s.FinalPosition.PieceCount > 15,

                                     GameEnginesCommon.CreateGameEngine(EngineType.CeresMCGS, "~T79", null),
                                     GameEnginesCommon.CreateGameEngine(EngineType.CeresMCGS2, "~T79", null),
                                     GameEnginesCommon.CreateGameEngine(EngineType.LC0_DAG_CUDA,null,  "~T79"),

                                     SearchLimit.NodesPerMove(10_000), // search limit
                                     [0],//new int[] { 0, 1, 2, 3 },
                                     MCGSTest.SEARCH_PARAMS_MCGS, MCGSTest.SELECT_PARAMS_MCGS,
                                     MCGSTest.SEARCH_PARAMS_MCGS2, MCGSTest.SELECT_PARAMS_MCGS2,
                                     true, // verbose                                    
                                     1, // engine 1 limit multiplier
                                     20,// engine 2 limit multiplier
                                     false, // Stockfish crosscheck
                                     null, // result callback
                                     0.25f); // Q-diff threshold
    CompareEngineResultSummary result = new CompareEnginesVersusOptimal(parms).Run();


  }

}
