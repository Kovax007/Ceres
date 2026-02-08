#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres.  If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives 

using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Ceres.APIExamples;
using Ceres.Base.Benchmarking;
using Ceres.Base.CUDA;
using Ceres.Base.Environment;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Commands;
using Ceres.Features;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Tournaments;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Params;
using Chess.Ceres.PlayEvaluation;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

#endregion

namespace Ceres
{

    public class TournamentConfig
    {
        public string CeresJsonPath { get; set; }
        public string CeresNetPath { get; set; }
        public int NumGamePairs { get; set; }
        public string OpeningsFile { get; set; }
        public int ConcurrentGames { get; set; }
        public int[] GPUIndices { get; set; }
        public string Device { get; set; }
        public int Engine1NodesPerMove { get; set; }
        public int Engine2NodesPerMove { get; set; }
        public string Engine2ExePath { get; set; }
        public Dictionary<string, double> SearchParams { get; set; }
    }

    public static class SPSATournamentRunner
    {
        // ========================================================================================================================
        const string CERES_JSON_PATH = @"C:\Lc0\Ceres\artifacts\release\net8.0\Ceres.json";

        const string CERES_NET_PATH = @"C:\Lc0\Ceres\artifacts\release\net8.0\C1-640-34.onnx";
        const string CERES_DEVICE = "GPU:0#TensorRT16"; // or #TensorRT16
        static int[] CONCURRENT_GAME_GPU_IDS = [0];

        const string SF_EXE_PATH = @"D:\Engines\Stockfish\src\stockfish.exe";
        const int SF_THREADS = 1;
        const int SF_TB_SIZE_MB = 16;
        const int SF_NODES_PER_MOVE = 300_000;

        const int CERES_NODES_PER_MOVE = 256;
        const int NUM_GAME_PAIRS = 2;
        const string OPENING_FN = @"C:\Lc0\books\UHO_Lichess_4852_v1.epd";
        const int NUM_CONCURRENT_GAMES = 4;
        // ========================================================================================================================

        public static void RunTournament(string configPath = null)
        {
            TournamentConfig config = null;

            if (configPath != null && File.Exists(configPath))
            {
                try
                {
                    string jsonText = System.IO.File.ReadAllText(configPath);
                    JsonSerializerOptions options = new JsonSerializerOptions { AllowTrailingCommas = true };
                    config = JsonSerializer.Deserialize<TournamentConfig>(jsonText, options);

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reading config file: {ex.Message}");
                    return;
                }
            }
            else
            {
                throw new ArgumentException($"No such file: {configPath}");
            }

            string CERES_JSON_PATH = config.CeresJsonPath;
            string CERES_NET_PATH = config.CeresNetPath;
            string CERES_DEVICE = config.Device;
            int CERES_NODES_PER_MOVE = config.Engine1NodesPerMove;
            string SF_EXE_PATH = config.Engine2ExePath;
            int SF_NODES_PER_MOVE = config.Engine2NodesPerMove;
            int[] CONCURRENT_GAME_GPU_IDS = config.GPUIndices;
            int NUM_GAME_PAIRS = config.NumGamePairs;
            string OPENING_FN = config.OpeningsFile;
            int NUM_CONCURRENT_GAMES = config.ConcurrentGames;

            CeresUserSettingsManager.LoadFromFile(CERES_JSON_PATH);
            string CERES_NETWORK = CeresUserSettingsManager.Settings.DefaultNetworkSpecString;
            string TB_DIR = CeresUserSettingsManager.Settings.DirTablebases;
            SearchLimit CERES_TIME_CONTROL = SearchLimit.NodesPerMove(CERES_NODES_PER_MOVE);
            SearchLimit SF_TIME_CONTROL = SearchLimit.NodesPerMove(SF_NODES_PER_MOVE);

            const string logfile = "ceres.log.txt"; //null;

            // Define Stockfish engine (via UCI) 
            GameEngineDefUCI sfEngine = new GameEngineDefUCI("SF", new GameEngineUCISpec("SF", SF_EXE_PATH, SF_THREADS, SF_TB_SIZE_MB, TB_DIR));

            // Turn on early search termination, and turn off overlapping executors (not needed for small searches).
            Ceres.MCTS.Params.ParamsSearch searchParams = new Ceres.MCTS.Params.ParamsSearch()
            {
                FutilityPruningStopSearchEnabled = true,
            };
            searchParams.Execution.FlowDirectOverlapped = false;
            searchParams.Execution.FlowDualSelectors = false;
            searchParams.Execution.NodeAnnotationCacheSize = 20_000;

            Ceres.MCTS.Params.ParamsSelect selectParams = new Ceres.MCTS.Params.ParamsSelect();

            // Alias map: Ceres.json config names → actual C# field names
            Dictionary<string, (Type targetType, string fieldName)> paramAliases = new()
            {
                ["PolicyTemperature"] = (typeof(Ceres.MCTS.Params.ParamsSelect), "PolicySoftmax"),
                ["FPU"] = (typeof(Ceres.MCTS.Params.ParamsSelect), "FPUValueAtRoot"),
                ["Contempt"] = (typeof(Ceres.MCTS.Params.ParamsSearch), "Contempt"),
            };

            // Apply search parameter overrides from config (for SPSA tuning)
            if (config.SearchParams != null)
            {
                foreach (var kvp in config.SearchParams)
                {
                    // Check alias map first
                    if (paramAliases.TryGetValue(kvp.Key, out var alias))
                    {
                        FieldInfo aliasField = alias.targetType.GetField(alias.fieldName, BindingFlags.Public | BindingFlags.Instance);
                        if (aliasField != null)
                        {
                            object target = alias.targetType == typeof(Ceres.MCTS.Params.ParamsSelect) ? selectParams : searchParams;
                            aliasField.SetValue(target, (float)kvp.Value);
                            Console.WriteLine($"Set {alias.targetType.Name}.{alias.fieldName} = {kvp.Value} (alias: {kvp.Key})");
                            continue;
                        }
                    }

                    // Try direct field name match on ParamsSelect, then ParamsSearch
                    FieldInfo selectField = typeof(Ceres.MCTS.Params.ParamsSelect).GetField(kvp.Key, BindingFlags.Public | BindingFlags.Instance);
                    if (selectField != null)
                    {
                        selectField.SetValue(selectParams, (float)kvp.Value);
                        Console.WriteLine($"Set ParamsSelect.{kvp.Key} = {kvp.Value}");
                    }
                    else
                    {
                        FieldInfo searchField = typeof(Ceres.MCTS.Params.ParamsSearch).GetField(kvp.Key, BindingFlags.Public | BindingFlags.Instance);
                        if (searchField != null)
                        {
                            searchField.SetValue(searchParams, (float)kvp.Value);
                            Console.WriteLine($"Set ParamsSearch.{kvp.Key} = {kvp.Value}");
                        }
                        else
                        {
                            Console.WriteLine($"WARNING: Unknown search parameter '{kvp.Key}', ignoring.");
                        }
                    }
                }
            }

            // Define Ceres engine (in process) with associated neural network and GPU and parameter customizations
            NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(CERES_NET_PATH, CERES_DEVICE);
            GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", ceresNNDef, null,
                                                                        searchParams,
                                                                        selectParams,
                                                                        logFileName: logfile);

            // Define players using these engines and specified time control
            EnginePlayerDef playerCeres = new EnginePlayerDef(engineDefCeres1, CERES_TIME_CONTROL);
            EnginePlayerDef playerSF = new EnginePlayerDef(sfEngine, SF_TIME_CONTROL);

            // Create a tournament definition
            TournamentDef tournDef = new TournamentDef("Ceres_vs_Stockfish", playerCeres, playerSF);
            tournDef.NumGamePairs = NUM_GAME_PAIRS;
            tournDef.OpeningsFileName = OPENING_FN;
            tournDef.ShowGameMoves = false;
            tournDef.AdjudicateMinNumMoves = int.MaxValue;
            tournDef.AdjudicateDrawThresholdNumMoves = int.MaxValue;
            tournDef.AdjudicateDrawThresholdCentipawns = int.MaxValue;
            tournDef.UseTablebasesForAdjudication = false;
            tournDef.OpeningRandomization = OpeningRandomizationEnum.Randomize;

            // Run the tournament
            TimingStats stats = new TimingStats();
            TournamentResultStats results;
            using (new TimingBlock(stats, TimingBlock.LoggingType.None))
            {
                results = new TournamentManager(tournDef, NUM_CONCURRENT_GAMES, CONCURRENT_GAME_GPU_IDS).RunTournament();
            }
            Console.WriteLine();
            Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");

            int indexCeres = results.Players[0].Name.ToUpper().Contains("CERES") ? 0 : 1;
            PlayerStat ceresResults = results.Players[indexCeres];
            PlayerStat sfResults = results.Players[indexCeres == 0 ? 1 : 0];
            float eloDiff = EloCalculator.EloDiff(ceresResults.PlayerWins, ceresResults.Draws, ceresResults.PlayerLosses);
            Console.WriteLine($"CERES W/D/L {ceresResults.PlayerWins} {ceresResults.Draws} {ceresResults.PlayerLosses}");

            // Compute pentanomial statistics by grouping game pairs by opening index.
            // Result in TournamentGameInfo is from Engine1's (Player1/Ceres) perspective.
            int ww = 0, wd = 0, wl = 0, dd = 0, ld = 0, ll = 0;
            var gamesByOpening = results.GameInfos
                .GroupBy(g => g.OpeningIndex)
                .OrderBy(g => g.Key);
            foreach (var pair in gamesByOpening)
            {
                var games = pair.OrderBy(g => g.GameSequenceNum).ToList();
                if (games.Count != 2) continue;

                // Result is from Engine1 (Ceres) perspective: Win=Ceres won, Loss=Ceres lost.
                int CeresResult(TournamentGameInfo g)
                {
                    if (g.Result == TournamentGameResult.Win) return 1;   // Ceres won
                    if (g.Result == TournamentGameResult.Loss) return -1;  // Ceres lost
                    return 0; // Draw
                }

                int r1 = CeresResult(games[0]);
                int r2 = CeresResult(games[1]);
                int ceresWins = (r1 == 1 ? 1 : 0) + (r2 == 1 ? 1 : 0);
                int ceresDraws = (r1 == 0 ? 1 : 0) + (r2 == 0 ? 1 : 0);
                int ceresLosses = (r1 == -1 ? 1 : 0) + (r2 == -1 ? 1 : 0);

                if (ceresWins == 2) ww++;
                else if (ceresWins == 1 && ceresDraws == 1) wd++;
                else if (ceresWins == 1 && ceresLosses == 1) wl++;
                else if (ceresDraws == 2) dd++;
                else if (ceresLosses == 1 && ceresDraws == 1) ld++;
                else if (ceresLosses == 2) ll++;
            }
            Console.WriteLine($"PENTANOMIAL {ww} {wd} {wl} {dd} {ld} {ll}");

            Console.WriteLine("ELO_DIFFERENCE " + eloDiff);
            System.Environment.Exit(0);
        }
    }

    public static class Program
    {
        /// <summary>
        /// Startup method for Ceres UCI chess engine and supplemental features.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            string configPath = null;

            if (args.Length == 0)
            {
                LaunchUCI(args);
                return;
            }

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--config" && i + 1 < args.Length)
                {
                    configPath = args[i + 1];
                    i++;
                }
                else
                {
                    Console.WriteLine($"Unknown argument: {args[i]}");
                    Console.WriteLine("Usage: Ceres.exe --config <path to config file>");
                    System.Environment.Exit(1);
                }
            }

            SPSATournamentRunner.RunTournament(configPath);
        }


        /// <summary>
        /// Perform engine initialization and enters into UCI processing loop.
        /// </summary>
        /// <param name="args"></param>
        /// <param name="searchModifier"></param>
        /// <param name="selectModifier"></param>
        /// <param name="enableLogging"></param>
        public static void LaunchUCI(string[] args,
                                     Action<ParamsSearch> searchModifier = null,
                                     Action<ParamsSelect> selectModifier = null,
                                     bool enableLogging = false)
        {
#if DEBUG
      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "*** WARNING: Ceres binaries built in Debug mode and will run much more slowly than Release");
#endif

            OutputBanner();
            CheckRecursiveOverflow();
            HardwareManager.VerifyHardwareSoftwareCompatability();

            // Load (or cause to be created) a settings file.
            if (!CeresUserSettingsManager.DefaultConfigFileExists)
            {
                const string DEFAULT_CERES_JSON = @"
{
  ""SyzygyPath"": null,
  ""DirCeresNetworks"": ""."",
  ""DirLC0Networks"": ""."",
  ""DefaultDeviceSpecString"": ""GPU:0"",
}
";

                Console.WriteLine();
                ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"*** NOTE: Configuration file {CeresUserSettingsManager.DefaultCeresConfigFileName} not found in working directory.");
                Console.WriteLine($"A new Ceres.json will be created with default settings:");
                Console.WriteLine(DEFAULT_CERES_JSON);

                System.IO.File.WriteAllText(CeresUserSettingsManager.DefaultCeresConfigFileName, DEFAULT_CERES_JSON);
                CeresUserSettingsManager.LoadFromDefaultFile();
            }

            // Configure logging level
            CeresEnvironment.MONITORING_EVENTS = enableLogging;
            LogLevel logLevel = enableLogging ? LogLevel.Information : LogLevel.Critical;
            LoggerTypes loggerTypes = LoggerTypes.WinDebugLogger | LoggerTypes.ConsoleLogger;
            CeresEnvironment.Initialize(loggerTypes, logLevel);

            CeresEnvironment.MONITORING_METRICS = !CommandLineWorkerSpecification.IsWorker
                                                 && CeresUserSettingsManager.Settings.LaunchMonitor;

            //      if (CeresUserSettingsManager.Settings.DirLC0Networks != null)
            //        NNWeightsFilesLC0.RegisterDirectory(CeresUserSettingsManager.Settings.DirLC0Networks);

            // Perform low-level hardware initialization.
            MCTSEngineInitialization.BaseInitialize(CeresEnvironment.MONITORING_METRICS, CeresUserSettingsManager.Settings.NUMANode);

            Console.WriteLine();

            //Features.BatchAnalysis.BatchAnalyzer.Test();      return;

            if (args != null && args.Length > 0 && (args[0].ToUpper() == "CUSTOM" || args[0].StartsWith("WORKER")))
            {
                TournamentTest.Test();
                //TournamentTest.TestSFLeela(0, true); return;
                //        SuiteTest.RunSuiteTest(); return;
            }

#if DEBUG
      CheckDebugAllowed();
#endif

            StringBuilder allArgs = new StringBuilder();
            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    allArgs.Append(args[i] + " ");
                }
            }

            string allArgsString = allArgs.ToString();

            DispatchCommands.ProcessCommand(allArgsString, searchModifier, selectModifier);


            //  Win32.WriteCrashdumpFile(@"d:\temp\dump.dmp");
        }


        /// <summary>
        /// Because Ceres runs much more slowly under Debug mode (at least 30%)
        /// this check verifies a debug bulid will not run unless explicitly
        /// requested in the options file or environment variables.
        /// </summary>
        private static void CheckDebugAllowed()
        {
            if (!CeresUserSettingsManager.Settings.DebugAllowed
              && Environment.GetEnvironmentVariable("CERES_DEBUG") == null)
            {
                const string MSG = "ERROR: Ceres was compiled in Debug mode and will only run\r\n"
                                 + "if the the DebugAllowed option is set to true\r\n"
                                 + "or the operating system environment variable CERES_DEBUG is defined.";
                Console.WriteLine();
                ConsoleUtils.WriteLineColored(ConsoleColor.Red, MSG);
                System.Environment.Exit(-1);
            }
        }


        const string BannerString =
        @"
|=========================================================|
| Ceres - A Monte Carlo Tree Search Chess Engine          |
|                                                         |
| (c) 2020- David Elliott and the Ceres Authors           |
|   Use help to list available commands.                  |
|                                                         |
|  Version {VER}                                       |
|  Runtime {VER}                                       |
|=========================================================|
";

        static void OutputBanner()
        {
            string dotnetVersion = RuntimeInformation.FrameworkDescription;
            (int majorCUDAVersion, int minorCUDAVersion) = CUDADevice.GetCUDAVersion();

            string cudaVersion = $"{majorCUDAVersion}.{minorCUDAVersion}";

            string[] bannerLines = BannerString.Split(Environment.NewLine);
            foreach (string line in bannerLines)
            {
                if (line.StartsWith("| Ceres"))
                {
                    ConsoleColor defaultColor = Console.ForegroundColor;
                    Console.Write("|");
                    Console.ForegroundColor = ConsoleColor.Magenta;
                    Console.Write(line.Substring(1, line.Length - 2));
                    Console.ForegroundColor = defaultColor;
                    Console.WriteLine("|");
                }

                else if (line.StartsWith("|  Version"))
                {
                    string version = $"|  Version {CeresVersion.VersionString} [git:{GitInfo.LastCommitSHA}]";
                    int spaceLeft = line.Length - version.Length;
                    string empty = new string(' ', 3 + spaceLeft - 1);
                    Console.WriteLine($"{version}{empty}|");
                }
                else if (line.StartsWith("|  Runtime"))
                {
                    string runtime = $"|  Runtime {dotnetVersion} and CUDA {cudaVersion}";
                    int spaceLeft = line.Length - runtime.Length;
                    string empty = new string(' ', 3 + spaceLeft - 1);
                    Console.WriteLine($"{runtime}{empty}|");
                }
                else
                {
                    Console.WriteLine(line);
                }
            }
        }



        /// <summary>
        /// Shuts down process if too many Ceres executables are running.
        /// This prevents situation where computer becomes unresponsive
        /// due to infinite cascade of Ceres processes (due to a coding error).
        /// </summary>
        static void CheckRecursiveOverflow()
        {
            int countCeres = 0;
            foreach (Process p in Process.GetProcesses())
            {
                if (p.ProcessName.ToUpper().Contains("CERES"))
                    countCeres++;
            }

            const int MAX_CERES_EXECUTABLE = 20;
            if (countCeres > MAX_CERES_EXECUTABLE)
            {
                Console.WriteLine("Shutting down, possible infinite process recursion, too many Ceres executables running running");
                System.Environment.Exit(3);
            }
        }

    }
}
