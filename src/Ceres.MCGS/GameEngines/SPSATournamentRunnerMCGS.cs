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
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Tournaments;
using Ceres.MCGS.Search.Params;
using Chess.Ceres.PlayEvaluation;

#endregion

namespace Ceres.MCGS.GameEngines;

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
    public string Engine2NetPath { get; set; }
    public int Engine2Threads { get; set; } = 1;
    public Dictionary<string, double> SearchParams { get; set; }
}

public static class SPSATournamentRunnerMCGS
{
    // ========================================================================================================================
    const string CERES_JSON_PATH = @"/home/privateclient/Ceres-spsa/artifacts/release/net10.0/Ceres.json";

    const string CERES_NET_PATH = @"/home/privateclient/Ceres-spsa/artifacts/release/net10.0/Ceres.MCGS";
    const string CERES_DEVICE = "GPU:0#TensorRTNative";
    static int[] CONCURRENT_GAME_GPU_IDS = [0];

    const string SF_EXE_PATH = @"/home/privateclient/spsa/Stockfish/src/stockfish";
    const int SF_THREADS = 1;
    const int SF_TB_SIZE_MB = 16;
    const int SF_NODES_PER_MOVE = 300_000;

    const int CERES_NODES_PER_MOVE = 256;
    const int NUM_GAME_PAIRS = 2;
    const string OPENING_FN = @"/home/privateclient/books/UHO_Lichess_4852_v1.epd";
    const int NUM_CONCURRENT_GAMES = 3;
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

        ParamsSearch searchParams = new ParamsSearch();
        ParamsSelect selectParams = new ParamsSelect();
        searchParams.Execution.DualOverlappedIterators = false;
        searchParams.Execution.DualEvaluators = false;

        // Alias map: tuner parameter names → one or more C# fields to set together
        Dictionary<string, List<(Type targetType, string fieldName)>> paramAliases = new()
        {
            ["CPUCT"] = new() {
                (typeof(ParamsSelect), "CPUCT"),
                (typeof(ParamsSelect), "CPUCTAtRoot"),
            },
            ["PolicyTemperature"] = new() {
                (typeof(ParamsSelect), "PolicySoftmax"),
            },
            ["FPU"] = new() {
                (typeof(ParamsSelect), "FPUValue"),
                (typeof(ParamsSelect), "FPUValueAtRoot"),
            },
        };

        // Apply search parameter overrides from config (for SPSA tuning)
        if (config.SearchParams != null)
        {
            foreach (var kvp in config.SearchParams)
            {
                // Check alias map first (supports setting multiple fields per parameter)
                if (paramAliases.TryGetValue(kvp.Key, out var aliasList))
                {
                    foreach (var alias in aliasList)
                    {
                        FieldInfo aliasField = alias.targetType.GetField(alias.fieldName, BindingFlags.Public | BindingFlags.Instance);
                        if (aliasField != null)
                        {
                            object target = alias.targetType == typeof(ParamsSelect) ? selectParams : searchParams;
                            aliasField.SetValue(target, (float)kvp.Value);
                            Console.WriteLine($"Set {alias.targetType.Name}.{alias.fieldName} = {kvp.Value} (alias: {kvp.Key})");
                        }
                    }
                    continue;
                }

                // Try direct field name match on ParamsSelect, then ParamsSearch
                FieldInfo selectField = typeof(ParamsSelect).GetField(kvp.Key, BindingFlags.Public | BindingFlags.Instance);
                if (selectField != null)
                {
                    selectField.SetValue(selectParams, (float)kvp.Value);
                    Console.WriteLine($"Set ParamsSelect.{kvp.Key} = {kvp.Value}");
                }
                else
                {
                    FieldInfo searchField = typeof(ParamsSearch).GetField(kvp.Key, BindingFlags.Public | BindingFlags.Instance);
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
        //searchParams.EnableGraph = false;
        searchParams.Execution.NNBatchSizeAlignmentTarget = 0;
        // Dump applied search parameters for verification
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(selectParams, new ParamsSelect(), false));
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(searchParams, new ParamsSearch(), false));
        Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(searchParams.Execution, new ParamsSearchExecution(), false));
        // Define MCGS (V2) engine with associated neural network, GPU, and parameter customizations
        NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(CERES_NET_PATH, CERES_DEVICE);
        GameEngineDefCeresMCGS engineDefCeres1 = new GameEngineDefCeresMCGS("CeresMCGS", ceresNNDef,
                                                                             searchParams, selectParams);

        // Define Engine1 (tested net) player
        EnginePlayerDef playerCeres = new EnginePlayerDef(engineDefCeres1, CERES_TIME_CONTROL);

        // Define Engine2 player: Ceres head-to-head (Engine2NetPath) or AB engine (Engine2ExePath)
        EnginePlayerDef playerEngine2;
        string tournamentName;
        SearchLimit engine2TimeControl = SearchLimit.NodesPerMove(SF_NODES_PER_MOVE);

        if (!string.IsNullOrEmpty(config.Engine2NetPath))
        {
            // Head-to-head: Ceres vs Ceres with different nets
            // Engine2ExePath and Engine2Threads are not used in this mode
            NNEvaluatorDef ceresNNDef2 = NNEvaluatorDefFactory.FromSpecification(config.Engine2NetPath, CERES_DEVICE);
            GameEngineDefCeresMCGS engineDefCeres2 = new GameEngineDefCeresMCGS("CeresBaseline", ceresNNDef2,
                                                                                 searchParams, selectParams);
            playerEngine2 = new EnginePlayerDef(engineDefCeres2, engine2TimeControl);
            tournamentName = "CeresMCGS_vs_CeresBaseline";
            Console.WriteLine($"Head-to-head mode: {CERES_NET_PATH} vs {config.Engine2NetPath}");
        }
        else
        {
            // AB engine mode (Stockfish, etc.) — uses Engine2ExePath and Engine2Threads
            int sfThreads = config.Engine2Threads;
            GameEngineDefUCI sfEngine = new GameEngineDefUCI("SF", new GameEngineUCISpec("SF", SF_EXE_PATH, sfThreads, SF_TB_SIZE_MB, TB_DIR));
            playerEngine2 = new EnginePlayerDef(sfEngine, engine2TimeControl);
            tournamentName = "CeresMCGS_vs_Stockfish";
        }

        // Create a tournament definition
        TournamentDef tournDef = new TournamentDef(tournamentName, playerCeres, playerEngine2);
        tournDef.NumGamePairs = NUM_GAME_PAIRS;
        tournDef.OpeningsFileName = OPENING_FN;
        tournDef.ShowGameMoves = false;
        tournDef.AdjudicateMinNumMoves = int.MaxValue;
        tournDef.AdjudicateDrawThresholdNumMoves = int.MaxValue;
        tournDef.AdjudicateDrawThresholdCentipawns = int.MaxValue;
        tournDef.UseTablebasesForAdjudication = false;
        tournDef.AdjudicateWinThresholdCentipawns = int.MaxValue;
        tournDef.AdjudicateWinThresholdNumMovesDecisive = int.MaxValue;
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

        // Player 0 is always Engine1 (tested net "CeresMCGS"), Player 1 is Engine2
        // For AB engines: fallback to name check in case tournament reorders players
        int indexCeres = results.Players[0].Name.Contains("CeresMCGS") ? 0
                       : results.Players[1].Name.Contains("CeresMCGS") ? 1
                       : 0; // default to first player
        PlayerStat ceresResults = results.Players[indexCeres];
        PlayerStat opponentResults = results.Players[indexCeres == 0 ? 1 : 0];
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
