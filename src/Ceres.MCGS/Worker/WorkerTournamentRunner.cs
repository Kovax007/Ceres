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
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Tournaments;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;
using Chess.Ceres.PlayEvaluation;

#endregion

namespace Ceres.MCGS.Worker;

/// <summary>
/// Runs a stoppable tournament and streams game-pair results back via a callback.
///
/// Uses the Ceres TournamentManager internally but wraps it with:
/// - A per-game-pair callback for streaming results
/// - Early stop via the Def.parentDef.ShouldShutDown flag
/// - Pentanomial computation from the accumulated results
/// </summary>
public class WorkerTournamentRunner
{
  /// <summary>
  /// Delegate invoked after each game pair completes (from the game thread).
  /// </summary>
  public delegate void GamePairCompletedHandler(GamePairResult result);

  /// <summary>
  /// The Ceres engine's network spec string (engine path + options).
  /// Updated when weights are refitted to point to the same loaded engine.
  /// </summary>
  private readonly string _ceresNetPath;

  private readonly string _ceresJsonPath;
  private readonly string _opponentExe;
  private readonly int _opponentNodes;
  private readonly int _opponentThreads;
  private readonly int _engineNodes;
  private readonly int _gpuId;
  private readonly string _bookPath;
  private readonly Dictionary<string, double> _searchParams;

  // Running tournament state
  private TournamentDef _currentDef;
  private string _currentPerturbationId;
  private int _wins, _draws, _losses;
  private readonly object _statsLock = new();

  // Pentanomial tracking: key=openingIndex, value=(r1, r2) from Ceres perspective
  private readonly Dictionary<int, (int r1, int r2)> _gamePairsByOpening = new();


  public WorkerTournamentRunner(
      string ceresNetPath,
      string ceresJsonPath,
      string opponentExe,
      int opponentNodes,
      int opponentThreads,
      int engineNodes,
      int gpuId,
      string bookPath,
      Dictionary<string, double> searchParams)
  {
    _ceresNetPath = ceresNetPath;
    _ceresJsonPath = ceresJsonPath;
    _opponentExe = opponentExe;
    _opponentNodes = opponentNodes;
    _opponentThreads = opponentThreads;
    _engineNodes = engineNodes;
    _gpuId = gpuId;
    _bookPath = bookPath;
    _searchParams = searchParams ?? new();
  }


  /// <summary>
  /// Run a tournament with the currently loaded (refitted) engine.
  /// Streams game-pair results via the callback.
  /// Returns the final TournamentResult when complete or stopped.
  /// </summary>
  public async Task<TournamentResult> RunAsync(
      PlayConfig playConfig,
      GamePairCompletedHandler onGamePairCompleted,
      CancellationToken ct = default)
  {
    _currentPerturbationId = playConfig.PerturbationId;
    _wins = 0;
    _draws = 0;
    _losses = 0;
    _gamePairsByOpening.Clear();

    // Load Ceres settings
    CeresUserSettingsManager.LoadFromFile(_ceresJsonPath);
    string tbDir = CeresUserSettingsManager.Settings.DirTablebases;

    // Configure search params
    var searchParams = new ParamsSearch();
    var selectParams = new ParamsSelect();
    searchParams.Execution.DualOverlappedIterators = false;
    searchParams.Execution.DualEvaluators = false;
    searchParams.Execution.NNBatchSizeAlignmentTarget = 0;
    ApplySearchParams(searchParams, selectParams, _searchParams);

    // Engine definitions
    string device = $"GPU:{_gpuId}#TensorRTNative";
    NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(_ceresNetPath, device);
    GameEngineDefCeresMCGS engineDefCeres = new("CeresMCGS", ceresNNDef, searchParams, selectParams);
    EnginePlayerDef playerCeres = new(engineDefCeres, SearchLimit.NodesPerMove(_engineNodes));

    // Opponent (Stockfish)
    GameEngineDefUCI sfEngine = new("SF", new GameEngineUCISpec("SF", _opponentExe,
        _opponentThreads, 16, tbDir));
    EnginePlayerDef playerSF = new(sfEngine, SearchLimit.NodesPerMove(_opponentNodes));

    // Tournament definition
    TournamentDef def = new("Worker_" + _currentPerturbationId, playerCeres, playerSF);
    def.NumGamePairs = playConfig.NumGamePairs;
    def.OpeningsFileName = _bookPath;
    def.ShowGameMoves = false;
    def.AdjudicateMinNumMoves = int.MaxValue;
    def.AdjudicateDrawThresholdNumMoves = int.MaxValue;
    def.AdjudicateDrawThresholdCentipawns = int.MaxValue;
    def.UseTablebasesForAdjudication = false;
    def.AdjudicateWinThresholdCentipawns = int.MaxValue;
    def.AdjudicateWinThresholdNumMovesDecisive = int.MaxValue;
    def.OpeningRandomization = OpeningRandomizationEnum.Randomize;
    _currentDef = def;

    // Register cancellation → sets ShouldShutDown flag
    ct.Register(() =>
    {
      if (_currentDef != null)
      {
        // parentDef is set when RunTournament clones the def for each thread
        // Before cloning, def IS the parent
        _currentDef.ShouldShutDown = true;
      }
    });

    // Run tournament on a thread pool thread
    TournamentResultStats results = await Task.Run(() =>
    {
      int concurrency = Math.Max(1, playConfig.Concurrency);
      int[] gpuIds = new[] { _gpuId };
      var mgr = new TournamentManager(def, concurrency, gpuIds);
      return mgr.RunTournament(enableCancelVialCtrlC: false);
    }, ct);

    // Process results
    ProcessResults(results, onGamePairCompleted);

    // Compute pentanomial
    int[] pentanomial = ComputePentanomial();

    return new TournamentResult
    {
      Type = ct.IsCancellationRequested ? "stopped" : "tournament_done",
      PerturbationId = _currentPerturbationId,
      Wins = _wins,
      Draws = _draws,
      Losses = _losses,
      GamesPlayed = _wins + _draws + _losses,
      Pentanomial = pentanomial
    };
  }


  /// <summary>
  /// Run a Ceres-vs-Ceres tournament between two networks.
  /// Both engines use the same search params and nodes per move.
  /// Results are from Engine 1's perspective (net1 = "our" engine).
  /// </summary>
  public async Task<TournamentResult> RunNetVsNetAsync(
      NetVsNetConfig config,
      GamePairCompletedHandler onGamePairCompleted,
      CancellationToken ct = default)
  {
    _currentPerturbationId = "netvsnet";
    _wins = 0;
    _draws = 0;
    _losses = 0;
    _gamePairsByOpening.Clear();

    // Load Ceres settings (for tablebase dir, etc.)
    // If already loaded from a prior INIT, CeresUserSettingsManager keeps the state.
    // The book path comes from the worker's stored _bookPath if available,
    // otherwise the tournament runs without a book (which is unusual but allowed).
    string tbDir = CeresUserSettingsManager.Settings?.DirTablebases ?? "";

    // Configure search params (shared by both engines)
    var searchParams = new ParamsSearch();
    var selectParams = new ParamsSelect();
    searchParams.Execution.DualOverlappedIterators = false;
    searchParams.Execution.DualEvaluators = false;
    searchParams.Execution.NNBatchSizeAlignmentTarget = 0;
    ApplySearchParams(searchParams, selectParams, config.SearchParams ?? new());

    // Build net spec strings
    string net1Spec = config.Net1Prefix + config.Net1Path;
    if (!string.IsNullOrEmpty(config.Net1Options))
    {
      net1Spec += "|" + config.Net1Options;
    }

    string net2Spec = config.Net2Prefix + config.Net2Path;
    if (!string.IsNullOrEmpty(config.Net2Options))
    {
      net2Spec += "|" + config.Net2Options;
    }

    // Engine definitions — both are CeresMCGS
    string device = $"GPU:{_gpuId}#TensorRTNative";
    NNEvaluatorDef nn1Def = NNEvaluatorDefFactory.FromSpecification(net1Spec, device);
    NNEvaluatorDef nn2Def = NNEvaluatorDefFactory.FromSpecification(net2Spec, device);

    GameEngineDefCeresMCGS engineDef1 = new("CeresMCGS-1", nn1Def, searchParams, selectParams);
    GameEngineDefCeresMCGS engineDef2 = new("CeresMCGS-2", nn2Def, searchParams, selectParams);

    EnginePlayerDef player1 = new(engineDef1, SearchLimit.NodesPerMove(config.NodesPerMove));
    EnginePlayerDef player2 = new(engineDef2, SearchLimit.NodesPerMove(config.NodesPerMove));

    // Tournament definition
    TournamentDef def = new("NetVsNet", player1, player2);
    def.NumGamePairs = config.NumGamePairs;
    def.OpeningsFileName = _bookPath;
    def.ShowGameMoves = false;
    def.AdjudicateMinNumMoves = int.MaxValue;
    def.AdjudicateDrawThresholdNumMoves = int.MaxValue;
    def.AdjudicateDrawThresholdCentipawns = int.MaxValue;
    def.UseTablebasesForAdjudication = false;
    def.AdjudicateWinThresholdCentipawns = int.MaxValue;
    def.AdjudicateWinThresholdNumMovesDecisive = int.MaxValue;
    def.OpeningRandomization = OpeningRandomizationEnum.Randomize;
    _currentDef = def;

    // Register cancellation
    ct.Register(() =>
    {
      if (_currentDef != null)
      {
        _currentDef.ShouldShutDown = true;
      }
    });

    // Run tournament on a thread pool thread
    TournamentResultStats results = await Task.Run(() =>
    {
      int concurrency = Math.Max(1, config.Concurrency);
      // DeviceIDs passed to TournamentManager are additive offsets applied to the base
      // device index already embedded in the NNEvaluatorDef (GPU:_gpuId#TensorRTNative).
      // Pass [0] so TryModifyDeviceID(base + 0) = base — all threads stay on this GPU.
      int[] gpuIds = new[] { 0 };
      var mgr = new TournamentManager(def, concurrency, gpuIds);
      return mgr.RunTournament(enableCancelVialCtrlC: false);
    }, ct);

    // Process results — engine 1 ("CeresMCGS-1") is our reference player
    ProcessNetVsNetResults(results, onGamePairCompleted);

    // Compute pentanomial
    int[] pentanomial = ComputePentanomial();

    return new TournamentResult
    {
      Type = ct.IsCancellationRequested ? "stopped" : "tournament_done",
      PerturbationId = "netvsnet",
      Wins = _wins,
      Draws = _draws,
      Losses = _losses,
      GamesPlayed = _wins + _draws + _losses,
      Pentanomial = pentanomial
    };
  }


  /// <summary>
  /// Process results for a net-vs-net tournament.
  /// Results are from Engine 1 ("CeresMCGS-1") perspective.
  /// </summary>
  private void ProcessNetVsNetResults(TournamentResultStats results, GamePairCompletedHandler callback)
  {
    if (results == null) return;

    // Find Engine 1 player index by matching "CeresMCGS-1"
    int indexEngine1 = 0;
    if (results.Players.Count >= 2)
    {
      indexEngine1 = results.Players[0].Name.Contains("CeresMCGS-1") ? 0
                   : results.Players[1].Name.Contains("CeresMCGS-1") ? 1
                   : 0;
    }

    // Group by opening to reconstruct game pairs
    var gamesByOpening = results.GameInfos
        .GroupBy(g => g.OpeningIndex)
        .OrderBy(g => g.Key);

    foreach (var pair in gamesByOpening)
    {
      var games = pair.OrderBy(g => g.GameSequenceNum).ToList();
      if (games.Count != 2) continue;

      int r1 = CeresResult(games[0], indexEngine1);
      int r2 = CeresResult(games[1], indexEngine1);

      lock (_statsLock)
      {
        if (r1 == 1) _wins++; else if (r1 == -1) _losses++; else _draws++;
        if (r2 == 1) _wins++; else if (r2 == -1) _losses++; else _draws++;

        _gamePairsByOpening[pair.Key] = (r1, r2);
      }

      callback?.Invoke(new GamePairResult
      {
        PerturbationId = "netvsnet",
        OpeningIdx = pair.Key,
        R1 = r1,
        R2 = r2,
        CumulativeWDL = new[] { _wins, _draws, _losses }
      });
    }
  }


  /// <summary>
  /// Stop the current tournament gracefully.
  /// </summary>
  public void Stop()
  {
    if (_currentDef != null)
    {
      _currentDef.ShouldShutDown = true;
    }
  }


  /// <summary>
  /// Get current W/D/L and games played.
  /// </summary>
  public (int wins, int draws, int losses) GetCurrentWDL()
  {
    lock (_statsLock)
    {
      return (_wins, _draws, _losses);
    }
  }


  /// <summary>
  /// Process all game results from the completed tournament.
  /// Updates W/D/L and invokes callbacks for each game pair.
  /// </summary>
  private void ProcessResults(TournamentResultStats results, GamePairCompletedHandler callback)
  {
    if (results == null) return;

    // Find Ceres player index
    int indexCeres = 0;
    if (results.Players.Count >= 2)
    {
      indexCeres = results.Players[0].Name.Contains("CeresMCGS") ? 0
                 : results.Players[1].Name.Contains("CeresMCGS") ? 1
                 : 0;
    }

    // Group by opening to reconstruct game pairs
    var gamesByOpening = results.GameInfos
        .GroupBy(g => g.OpeningIndex)
        .OrderBy(g => g.Key);

    foreach (var pair in gamesByOpening)
    {
      var games = pair.OrderBy(g => g.GameSequenceNum).ToList();
      if (games.Count != 2) continue;

      int r1 = CeresResult(games[0], indexCeres);
      int r2 = CeresResult(games[1], indexCeres);

      lock (_statsLock)
      {
        // Update W/D/L
        if (r1 == 1) _wins++; else if (r1 == -1) _losses++; else _draws++;
        if (r2 == 1) _wins++; else if (r2 == -1) _losses++; else _draws++;

        // Track for pentanomial
        _gamePairsByOpening[pair.Key] = (r1, r2);
      }

      // Stream callback
      callback?.Invoke(new GamePairResult
      {
        PerturbationId = _currentPerturbationId,
        OpeningIdx = pair.Key,
        R1 = r1,
        R2 = r2,
        CumulativeWDL = new[] { _wins, _draws, _losses }
      });
    }
  }


  /// <summary>
  /// Extract Ceres result from a game: +1=win, 0=draw, -1=loss.
  /// </summary>
  private static int CeresResult(TournamentGameInfo game, int ceresPlayerIndex)
  {
    // Result is always from Player 0's perspective in TournamentGameResult
    // If Ceres is Player 0: Win=+1, Loss=-1
    // If Ceres is Player 1: Win=-1, Loss=+1 (inverted)
    if (ceresPlayerIndex == 0)
    {
      return game.Result switch
      {
        TournamentGameResult.Win => 1,
        TournamentGameResult.Loss => -1,
        _ => 0
      };
    }
    else
    {
      return game.Result switch
      {
        TournamentGameResult.Win => -1,
        TournamentGameResult.Loss => 1,
        _ => 0
      };
    }
  }


  /// <summary>
  /// Compute pentanomial statistics from accumulated game pairs.
  /// Returns [WW, WD, WL, DD, LD, LL].
  /// </summary>
  private int[] ComputePentanomial()
  {
    int ww = 0, wd = 0, wl = 0, dd = 0, ld = 0, ll = 0;

    lock (_statsLock)
    {
      foreach (var (r1, r2) in _gamePairsByOpening.Values)
      {
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
    }

    return new[] { ww, wd, wl, dd, ld, ll };
  }


  /// <summary>
  /// Apply search parameter overrides using the same alias map as SPSATournamentRunnerMCGS.
  /// </summary>
  private static void ApplySearchParams(ParamsSearch searchParams, ParamsSelect selectParams,
      Dictionary<string, double> overrides)
  {
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

    foreach (var kvp in overrides)
    {
      if (paramAliases.TryGetValue(kvp.Key, out var aliasList))
      {
        foreach (var alias in aliasList)
        {
          FieldInfo field = alias.targetType.GetField(alias.fieldName,
              BindingFlags.Public | BindingFlags.Instance);
          if (field != null)
          {
            object target = alias.targetType == typeof(ParamsSelect) ? selectParams : searchParams;
            field.SetValue(target, (float)kvp.Value);
          }
        }
        continue;
      }

      FieldInfo selectField = typeof(ParamsSelect).GetField(kvp.Key,
          BindingFlags.Public | BindingFlags.Instance);
      if (selectField != null)
      {
        selectField.SetValue(selectParams, (float)kvp.Value);
      }
      else
      {
        FieldInfo searchField = typeof(ParamsSearch).GetField(kvp.Key,
            BindingFlags.Public | BindingFlags.Instance);
        if (searchField != null)
        {
          searchField.SetValue(searchParams, (float)kvp.Value);
        }
      }
    }
  }
}
