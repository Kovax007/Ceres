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

using System.Collections.Generic;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.MCGS.Features.Suites;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Test;


#endregion

namespace Ceres.MCGS.Tests
{
  public static class MCGSSuiteTest
  {
    /// <summary>
    /// Experimental sample code of runnings suites via the API.
    /// </summary>
    public static void RunSuiteTest()
    {
      const int PARALLELISM = 1;
      //const string NET = "C1-256-10-i8|cudagraphs=false";
      //const string NET = "~T81";
      //      const string NET_CERES = "~T3_DISTILL_512_15_FP16_TRT|cudagraphs=false";
      //      const string NET_LC0 = "~T3_DISTILL_512_15_NATIVE";

      string NET_CERES1 = "~T1_DISTILL_256_10_FP16_TRT|cudagraphs=false";
      string NET_LC0 = null;// @"ONNX_CUDA:d:\nets\t1-256x10-distilled-swa-2432500.pb.gz";

      //      string NET_CERES1 = "~T3_DISTILL_512_15_FP16_TRT|cudagraphs=false";
      //      string NET_LC0 = "~T3_DISTILL_512_15_NATIVE";

      //string NET_CERES1 = "~T81";// "C1-512-15-i8|cudagraphs=true";
      string NET_CERES2 = "~T81";//"C1-512-15-i8|cudagraphs=true";
                                 //string NET_LC0 = "~T79";

      //NET_CERES1 = NET_CERES2 = "C1-512-15-i8";

      //      NET_CERES1 = NET_CERES2 = "C1-640-34-i8";
      //NET_CERES1 = NET_CERES2 = "~T3_DISTILL_512_15_FP16_TRT|cudagraphs=false";

      NET_CERES1 = "C1-384-12-i8|cudagraphs=true;V1FRAC=0.2;V1TEMP=0.50;V2TEMP=1.25;BLUN_NEG=0.05;BLUN_POS=0.05";
      NET_CERES2 = "C1-384-12-i8|cudagraphs=true";

      const string DEVICE = "GPU:0#TensorRT16";

      string deviceSuffix = PARALLELISM > 1 ? ":POOLED" : "";

      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(NET_CERES1, DEVICE);

      SearchLimit limit = SearchLimit.SecondsPerMove(3f);
      limit = SearchLimit.NodesPerMove(20_000);

      List<string> extraUCI = null; // new string[] { "setoption name Contempt value 5000" };

//      TestPositionMultipleEngines tpme = new TestPositionMultipleEngines(NET_CERES1, NET_LC0);
      //      tpme.Init(TestEngines.CeresMCGS | TestEngines.CeresMCTS);
      GameEngineDef lc0DAG = NET_LC0 == null ? null : MCGSTest.GameEngineLc0(NET_LC0, "GPU:0", MCGSTest.LC0EngineType.RewriteDAG, true);

      GameEngineDef ged1 = new GameEngineDefCeresMCGS("CeresMCGS", evalDef1,
                                                      MCGSTest.SEARCH_PARAMS_MCGS,
                                                      MCGSTest.SELECT_PARAMS_MCGS
#if NOT
                                                      MCGSTest.SEARCH_PARAMS_MCGS_COMMON with
                                                      {
                                                        //OffPathBackupNumAdditionalLevelsToPropagate = 5,
                                                        //EnablePseudoTranspositionBlending = false,
//                                                        VisitSuboptimalityRejectThreshold = 0.07f,
                                                        PathTranspositionMode = MCGSPathMode.Coalesce,
                                                        EnablePseudoTranspositionBlending = false,
                                                        //TestFlag2 = true,

                                                        //PathTranspositionMode = MCGSPathMode.Coalesce,
                                                        Execution = new Search.Params.ParamsSearchExecution() with
                                                        { 
                                                          //BackupMode  = Search.Params.BackupMethodEnum.ReductionSingleThread 
                                                        }
                                                        //TestFlag=true,
                                                      },
                                                      MCGSTest.SELECT_PARAMS_MCGS with
                                                      {
                                                        //CPUCT =2.32f,
                                                        //CPUCTAtRoot = 2.32f,
                                                        //FPUValue = 0.65f
                                                      }
#endif
                                                      )
      {

      };
      GameEngineDef ged2 =  new GameEngineDefCeresMCGS("CeresOther", evalDef1,
                                                      MCGSTest.SEARCH_PARAMS_MCGS2,
                                                      MCGSTest.SELECT_PARAMS_MCGS2

#if NOT
                                                      MCGSTest.SEARCH_PARAMS_MCGS_COMMON with
                                                      {
                                                        //EnablePseudoTranspositionBlending = false
                                                        //EnableGraph = false,
                                                      },
                                                      MCGSTest.SELECT_PARAMS_MCGS
#endif
                                                      );

      //      GameEngineUCISpec geSF = new GameEngineUCISpec("SF12", @"\\synology\dev\chess\engines\stockfish_20090216_x64_avx2.exe",
      //                                                     32, 2048, CeresUserSettingsManager.Settings.TablebaseDirectory,
      //                                                     uciSetOptionCommands: extraUCI);
      GameEngineDefUCI geOther = default;// TestPositionMultipleEngines.GameEngineCeresUCI(NET);

      EnginePlayerDef ceresEngineDef1 = new EnginePlayerDef(ged1, limit);
      EnginePlayerDef ceresEngineDef2 = new EnginePlayerDef(ged2, limit);

//      GameEngineDefUCI sf12EngineDef = new GameEngineDefUCI("SF12", geSF);
      EnginePlayerDef lc0EngineDef = lc0DAG == null ? null : new EnginePlayerDef(lc0DAG, limit);// * 875);

      string DIR = SoftwareManager.IsWindows ? @"\\synology\dev\chess\data\epd\" 
                                             : "/mnt/syndev/chess/data/epd/";

      const bool USE_LC0 = false;
      SuiteTestDef def = new SuiteTestDef("Test1",
        DIR + "hard-talkchess-2022.epd",
                                          //        @"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
                                          ceresEngineDef1,
                                          !USE_LC0 ? ceresEngineDef2 : null,
                                          USE_LC0 ? lc0EngineDef : null
                                          );
      def.MaxNumPositions = 3000;

      SuiteTestRunner ser = new SuiteTestRunner(def);
      ser.Run(PARALLELISM, true);
    }

  }
}
