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
using System.Collections.Generic;
using Ceres.Chess;
using Ceres.Chess.UserSettings;

using Ceres.Base.Benchmarking;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

using Ceres.MCGS.MCTSNodes.NEW.Test;
using Ceres.MCGS.MCTSNodes.Storage.Parents;

using CeresTrain.Examples;
using CeresTrain.NNEvaluators;
using CeresTrain.Trainer;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.MoveGen;

using Ceres.MCGS.MCTSNodes.NEW;

using Ceres.Features.GameEngines;


using Ceres.Chess.NNEvaluators;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.GameEngines;
using Ceres.MCGS.Graphs.Structs;
using Ceres.MCGS.Graphs.Wrappers;

#endregion

namespace Ceres.Train
{
  public static partial class MCGSTest
  {
    public static void DumpForChild(VisitFromStore table, int parentIndex)
    {
      Console.WriteLine("Dumping parents for entry " + parentIndex);
      Span<int> parents = stackalloc int[50];
      table.GetVisitsFrom(new NodeIndex(parentIndex), parents);
      for (int i = 0; i < parents.Length && parents[i] != -1; i++)
      {
        Console.WriteLine($"  Parent for {parentIndex} = {parents[i]}");
      }
    }


    public static void DoTest()
    {
#if NOT
      if (false)
      {
        NNEvaluatorTorchsharp evalCeres1 = (NNEvaluatorTorchsharp)NNEvaluatorDef.FromSpecification(evalSpec1, "GPU:0").ToEvaluator();

        // NOTE: this test flawed, is feeding in state for CURRENT position, not prior position!
        //        const string FEN = "r4r1k/4p1bp/2ppQ1p1/1p1nP3/p1nP1P2/1R1B4/q1P1N1P1/2B2RK1 w - - 2 22";//
        const string FEN = "1r3rk1/p4p1p/3pbnpQ/qBp5/4P2P/2N2P2/PPP3P1/2KR3R w - - 1 16";
        Position TEST_POS = Position.FromFEN(FEN);
        Console.WriteLine("starting pos");
        NNEvaluatorResult evalResultNoState = evalCeres1.Evaluate(TEST_POS);
        Console.WriteLine("no state          " + evalResultNoState);
        evalResultNoState = evalCeres1.Evaluate(TEST_POS);
        Console.WriteLine("no state (repeat) " + evalResultNoState);
        NNEvaluatorResult evalResultState = evalCeres1.Evaluate(TEST_POS, state: evalResultNoState.PriorState);
        //        evalResultState = evalCeres.Evaluate(TEST_POS, state: evalResultState.PriorState); // Run second time to pick up state
        Console.WriteLine("w/ state          " + evalResultState);
        evalResultState = evalCeres1.Evaluate(TEST_POS, state: evalResultNoState.PriorState);
        Console.WriteLine("w/ state (repeat) " + evalResultState);

        foreach (float f in new float[] { 0.15f, 0.30f, 0.45f })
        {
          evalCeres1.Options = evalCeres1.Options with { /*QNegativeBlunders = f*/ };
          evalResultState = evalCeres1.Evaluate(TEST_POS, state: null);
          Console.WriteLine("w/ blunder down  " + f + " " + evalResultState);

        }
        EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(2, Chess.NNEvaluators.NNEvaluator.InputTypes.All);
        builder.Add(TEST_POS);
        NNEvaluatorResult evalResultBatchNoState = evalCeres1.EvaluateBatch(builder.GetBatch())[0];
        Console.WriteLine("no state (batch) " + evalResultNoState);
        builder.ResetBatch();
        builder.Add(TEST_POS, state: evalResultNoState.PriorState);
        NNEvaluatorResult evalResultBatch = evalCeres1.EvaluateBatch(builder.GetBatch())[0];
        Console.WriteLine("w/ state (batch) " + evalResultBatch);
        System.Environment.Exit(3);
      }
#endif
      if (false)
      {
        // TODO: This is a buglet, if using MCGS engine then starting from a drawn by repetition does not return any legal moves.
        string fenx = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        string moves = "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 b7b6 h2h4 d8d7 h1h3 c8a6 a2a3 b4c3 h3c3 a6f1 e1f1 g8e7 g1e2 b8c6 b2b4 e8c8 a3a4 c6b4 a4a5 c8b7 c1a3 b4c6 a3e7 c6e7 a5b6 a7b6 c3a3 d8a8 d1d3 h7h5 a3a6 e7c6 a6a8 h8a8 a1a8 b7a8 g2g3 a8b7 e2f4 g7g6 c2c3 c6e7 f4h3 e7f5 h3g5 f5h6 f1g2 d7e8 d3c2 e8b5 g5f7 h6f7 c2g6 f7d8 g6h5 b5d3 h5e8 d3e4 g2h2 b7c8 h4h5 e4f3 h5h6 f3f2 h2h3 f2f7 e8h8 c8d7 h6h7 f7h5 h3g2 h5e2 g2g1 e2d1 g1g2 d1c2 g2g1 c2c1 g1g2 c1d2 g2g1 d2e3 g1g2 e3e4 g2g1 d8f7 h8g7 e4e1 g1g2 e1e2 g2g1 e2d1 g1g2 d1e2";
        GameEngineCeresInProcess gec = default;// new GameEngineCeresInProcess("x", NNEvaluatorDef.FromSpecification("~T70", "GPU:0"));
        var xx = gec.Search(PositionWithHistory.FromFENAndMovesUCI(fenx, moves), SearchLimit.NodesPerMove(5));
        Console.WriteLine(xx);
        Console.WriteLine("exiting xxxyyy");
        System.Environment.Exit(3);

      }


      const string B4_384 = "Ceres:DGX_S_384_12_FFN4_H16_NLA_B4_6bn_fp16_5399986176.onnx|4BOARD"; // 5.0bn good
      NNEvaluatorDef evalDefMCGS1 = NNEvaluatorDef.FromSpecification(B4_384, "GPU:0");
      NNEvaluatorDef evalDefMCGS2 = NNEvaluatorDef.FromSpecification(B4_384, "GPU:0");

      bool ENABLE_STATE1 = true;
      bool ENABLE_STATE2 = false;

      if (true)
      {
        RunTournamentTest(evalDefMCGS1, ENABLE_STATE1, evalDefMCGS2, ENABLE_STATE2);
      }

      for (int i = 0; i < 1; i++)
      {
        TestStateUsefulness.TestImpactOfStateEnabledOnValueAccuracy();
        //        GraphTestingIntegration.GraphIntegrationTest();
      }


      System.Environment.Exit(3);
      //      Nodex.Test();      System.Environment.Exit(3);
      //      TournamentTest.Test(); System.Environment.Exit(3);

      //TestMCTSNodeParentsTable();return;
      CeresUserSettingsManager.Settings.EnableOverlappingExecutors = false;


      //      NNEvaluator evaluator = NNEvaluator.FromSpecification("703810", "GPU:0");
      string fen = "8/2p2k1p/ppPp1p2/5Pp1/2P1K1P1/7P/PP6/8 w - - 0 1"; // #665 Averbakh
      fen = "4k3/8/8/1p1p1p1p/pPpPpPpP/P1P1P1P1/8/4K3 w - - 0 1"; // locked, much space
                                                                  //      fen = "2k1b3/1p1p1p2/1P1P1P2/8/8/2p1p1p1/2P1P1P1/3B1K2 w - - 0 1"; // locked, only about 75 reachable positions
      fen = Position.StartPosition.FEN;
      //      fen = "8/3k4/8/8/8/4K3/2Q5/8 b - - 0 1"; // easy checkmate soon

      NNEvaluatorDef evalDef = NNEvaluatorDef.FromSpecification("RANDOM_NARROW:703810", "GPU:0");
      MCTS.Params.ParamsSearch paramsSearch = new MCTS.Params.ParamsSearch();
      paramsSearch.EnableTablebases = false;
      MCTS.Params.ParamsSelect paramsSelect = new MCTS.Params.ParamsSelect();
      paramsSearch.Execution.MaxBatchSize = 1;
      paramsSearch.Execution.SelectParallelEnabled = false;

      GameEngineCeresInProcess engine = new GameEngineCeresInProcess("CeresMCGS", evalDef, searchParams: paramsSearch);

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

          //          engineResult.Search.Manager.Context.Tree.Store.Validate(engineResult.Search.Manager.Context.Tree.TranspositionRoots);
        }
      }


      //      engineResult.Search.Manager.Context.Tree.Store.Validate(engineResult.Search.Manager.Context.Tree.TranspositionRoots);

      Console.WriteLine(engineResult);
      //      engineResult.Search.Manager.Context.Tree.Store.Dump(false);
      engineResult.Search.Manager.DumpFullInfo(Console.Out, fen);
      return;

      Dictionary<string, ulong> positions = new();
      var tree = engineResult.Search.Manager.Context.Tree;
      var store = tree.Store;
#if NOT
      if (false) // Regular search does not see transpositions properly?
    {
      for (int i = 1; i <= store.Nodes.NumUsedNodes; i++)
      {
        ref NodeStruct node = ref store.Nodes.nodes[i];
        MCTSNode nodex = tree.GetNode(node.Index);
        string nodeFEN = nodex.Annotation.Pos.FEN;
        Console.WriteLine($"{i} {nodeFEN} {nodex.StructRef.ZobristHash}");

        if (positions.ContainsKey(nodeFEN))
        {
          Console.WriteLine($"Duplicate FEN  with hash {positions[nodeFEN]} versus our hash {nodex.StructRef.ZobristHash}, in TranspsitionRoots: {tree.TranspositionRoots.Dictionary.ContainsKey(nodex.StructRef.ZobristHash)}");
        }
        positions[nodex.Annotation.Pos.FEN] = nodex.StructRef.ZobristHash;
      }
#endif
      return;

    }


    static Func<(GameEngineCeresMCGSInProcess engine, PositionWithHistory Pos, SearchLimit Limit), (MGMove, float, int)> MoveMaker =
     inp =>
     {
       (GameEngineCeresMCGSInProcess engine, PositionWithHistory pos, SearchLimit limit) = inp;

       if (limit.Type != SearchLimitType.NodesPerMove)
       {
         throw new Exception("Unsupported search limit");
       }
       if (limit.Value == 1)
       {
         throw new Exception("Not yet configured for policy search");
       }

       //        NNEvaluatorResult result = evalCeres.Evaluate(pos);
       //        MGMove bestMove = result.Policy.TopMove(pos.FinalPosition);

       const bool HAS_ACTION = false;
       //       const bool HAS_STATE = false;
       bool ENABLE_STATE = engine.ID.EndsWith("1");

       Graph graphX = new Graph(300, HAS_ACTION, ENABLE_STATE, engine.SearchParams.EnableMCGS, pos);
       NodeX rootX = graphX.RootNode;
       if (!rootX.IsRoot)
       {
         throw new Exception("Root node is not root");
       }



       const int EXTRA_NODES_FOR_MCGS = 0;
       NNEvaluator evaluator = engine.Evaluators.Evaluator1;
       const bool FORCE_NO_TABLEBASE_TERMINALS = false; // **** TO DO!!! Fill this in (came from MCGSManager in original code)
       GraphTestingIntegration.RunBatchSelectAndBackup(graphX, engine.EvaluatorDef, evaluator, 0,
                                                       (int)limit.Value + EXTRA_NODES_FOR_MCGS, 1,
                                                       FORCE_NO_TABLEBASE_TERMINALS,
                                                       ENABLE_STATE,
                                                       engine.SearchParams.ActionHeadSelectionWeight,
                                                       false);

       VisitToStruct bestChild = default;
       int bestChildIndex = 0;
       int childIndex = 0;
       float minQ = float.MaxValue;
       int maxN = int.MinValue;
       const bool USE_N = false;
       foreach (VisitToStruct child in rootX.VisitsTo)
       {
         if (USE_N)
         {
           if (child.N > maxN)
           {
             maxN = child.N;
             bestChild = child;
             bestChildIndex = childIndex;
           }
         }
         else
         {
           if (child.W / child.N < minQ)
           {
             minQ = (float)child.W / child.N;
             bestChild = child;
             bestChildIndex = childIndex;
           }
         }
         childIndex++;
       }

       NodeX bestMoveNode = new NodeX(graphX, bestChild.VisitNodeIndex);
       MGMove bestMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(bestMoveNode.NodeRef.PriorMove, pos.FinalPosition.ToMGPosition);

       float q = (float)-bestChild.W / bestChild.N;
       int n = rootX.NodeRef.N;
       graphX.Store.Dispose();
       return (bestMove, q, n);
     };


    static GameEngineDefCeresMCGS MakeMCGSGameDef(string engineName, NNEvaluatorDef evalDef,
                                                  float actionHeadSelectionWeight,
                                                  bool enableState,
                                                  bool enableMCGS)
    {

      //      GameEngineCeresMCGSInProcess engineMCGS1 = new GameEngineCeresMCGSInProcess(engineName, evalDef);
      //NNEvaluator evaluatorMCGS1 = evalDef.ToEvaluator();

      GameEngineDefCeresMCGS engineDefMCGS = new(engineName, evalDef, actionHeadSelectionWeight, enableMCGS);
      GameEngineDefCeresMCGS.MoveMaker = MoveMaker; // WARN: static
      return engineDefMCGS;
    }


    private static void RunTournamentTest(NNEvaluatorDef evalDefMCGS1, bool enableState1, NNEvaluatorDef evalDefMCGS2, bool enableState2)
    {
      //      GameEngineCeresMCGSInProcess engineMCGS1 = new GameEngineCeresMCGSInProcess("CeresMCGS", evalDefMCGS1);
      //      NNEvaluator evaluatorMCGS1 = evalDefMCGS1.ToEvaluator();

      float ENGINE1_ACTION_WEIGHT = 0.0f;
      float ENGINE2_ACTION_WEIGHT = 0.0f;

      const bool ENABLE_GRAPH_1 = false;
      const bool ENABLE_GRAPH_2 = false;

      SearchLimit limit = SearchLimit.NodesPerMove(2);
      GameEngineDefCeresMCGS engineMCGS1 = MakeMCGSGameDef("CeresMCGS1", evalDefMCGS1, ENGINE1_ACTION_WEIGHT, enableState: enableState1, ENABLE_GRAPH_1);
      GameEngine engine1 = engineMCGS1.CreateEngine();
      engine1.Warmup(100);

      GameEngineSearchResult searchResult = engine1.Search(PositionWithHistory.FromPosition(Position.StartPosition), limit);
      Console.WriteLine();
      Console.WriteLine("MCGS Search of " + limit);
      Console.WriteLine(searchResult);
      System.Environment.Exit(3);
      GameEngineDefCeresMCGS engineMCGS2 = MakeMCGSGameDef("CeresMCGS2", evalDefMCGS2, ENGINE2_ACTION_WEIGHT, enableState: enableState2, ENABLE_GRAPH_2);

      //      const int SEARCH_NUM_NODES = 100;

      if (false)
      {
        PositionWithHistory pwh = new PositionWithHistory(Position.FromFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 "));
        //        GameEngineSearchResult ss = engineMCGS1.Search(pwh, SearchLimit.NodesPerMove(SEARCH_NUM_NODES));
        //        Console.WriteLine(ss);
        //    public Func<(PositionWithHistory, SearchLimit), (string, float, int)> MoveMaker;
      }

      //      var s1 = engineMCGS1.CreateEngine().Search(PositionWithHistory.StartPosition, SearchLimit.NodesPerMove(2));
      //      var s2 = engineMCGS2.CreateEngine().Search(PositionWithHistory.StartPosition, SearchLimit.NodesPerMove(2));

      TPGNetTests.RunTournament(engineMCGS1, engineMCGS2);
      System.Environment.Exit(3);
    }
  }
}





#if NOT
    static MLPNetDef configTransformerNetDefLinear
  = new((4096, nn.ReLU()), (2048, nn.ReLU()), (1024, nn.ReLU()));


    /// <summary>
    /// Read and validate records from a TPG file.
    /// </summary>
    public static void TestTPGFileReader()
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "TestTPGFileReader");
      TPGFileReader readerTPG = new TPGFileReader(@"e:\cout\temp\OnePawnEndings_1.dat.zst", 1024 * 16);
      using (new TimingBlock("TestTPGFileReader"))
      {
        int batchCount = 0;

        foreach (TPGRecord[] batch in readerTPG.Enumerable)
        {
          Console.WriteLine("  batch " + batchCount + " size " + readerTPG.BatchSize);
          for (int i = 0; i < batch.Length; i++)
          {
            TPGRecordValidation.Validate(in batch[i]);
          }

          TPGRecord testRecord = batch[373];
          testRecord.DumpPositionWithHistoryInBrowser();
          System.Environment.Exit(3);

          if (++batchCount >= 5)
          {
            break;
          }
        }
      }

#endif
#endif