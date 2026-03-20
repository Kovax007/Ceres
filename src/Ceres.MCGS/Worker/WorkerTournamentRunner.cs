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
  /// Returns a Task so callers can await it — prevents concurrent stream writes.
  /// </summary>
  public delegate Task GamePairCompletedHandler(GamePairResult result);

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
  // Updated live by PerGamePairCallback during tournament.
  private readonly Dictionary<int, (int r1, int r2)> _gamePairsByOpening = new();

  // Live game count accessible during play (for STATUS queries)
  private volatile int _liveGamesPlayed;


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

    // Opening seed: >= 0 uses ShuffleDeterministic (antithetical pairs share seed for CRN),
    // -1 (default) uses Randomize for independent perturbations.
    if (playConfig.OpeningSeed >= 0)
    {
      def.OpeningRandomization = OpeningRandomizationEnum.ShuffleDeterministic;
      def.OpeningShuffleSeed = playConfig.OpeningSeed;
    }
    else
    {
      def.OpeningRandomization = OpeningRandomizationEnum.Randomize;
    }
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

    // Run tournament on a thread pool thread.
    // PerGamePairCallback fires live from game threads to keep STATUS up to date.
    _liveGamesPlayed = 0;
    try
    {
      await Task.Run(() =>
      {
        int concurrency = Math.Max(1, playConfig.Concurrency);
        // DeviceIDs passed to TournamentManager are additive offsets applied to the base
        // device index already embedded in the NNEvaluatorDef (GPU:_gpuId#TensorRTNative).
        // Pass [0] so TryModifyDeviceID(base + 0) = base — all threads stay on this GPU.
        int[] gpuIds = new[] { 0 };
        var mgr = new TournamentManager(def, concurrency, gpuIds);
        mgr.PerGamePairCallback = (gameInfo, gameReverseInfo) =>
        {
          int r1 = CeresResult(gameInfo, 0);
          int r2 = CeresResult(gameReverseInfo, 0);
          lock (_statsLock)
          {
            if (r1 == 1) _wins++; else if (r1 == -1) _losses++; else _draws++;
            if (r2 == 1) _wins++; else if (r2 == -1) _losses++; else _draws++;
            _gamePairsByOpening[gameInfo.OpeningIndex] = (r1, r2);
            _liveGamesPlayed = _wins + _draws + _losses;
          }
        };
        mgr.RunTournament(enableCancelVialCtrlC: false);
      }, ct);
    }
    catch (OperationCanceledException)
    {
      throw;  // Propagate — expected when STOP arrives before Task.Run starts
    }
    catch (Exception ex)
    {
      // Non-fatal: TournamentResultStats display (DumpTournamentSummary) sometimes throws
      // when tournament stops early with few results (column width arithmetic bug).
      // The live W/D/L and _gamePairsByOpening are already populated — we can continue.
      Console.Error.WriteLine($"[WorkerTournamentRunner] RunTournament warning (non-fatal): {ex.Message}");
    }

    // Stream game-pair results to orchestrator (batch delivery after tournament).
    // W/D/L and _gamePairsByOpening were already updated live by PerGamePairCallback.
    await StreamResultsAsync(onGamePairCompleted);

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
  /// Get live W/D/L (updated during tournament via PerGamePairCallback).
  /// </summary>
  public (int wins, int draws, int losses) GetCurrentWDL()
  {
    lock (_statsLock)
    {
      return (_wins, _draws, _losses);
    }
  }

  /// <summary>
  /// Get live game count (updated during tournament).
  /// </summary>
  public int GetLiveGamesPlayed() => _liveGamesPlayed;

  /// <summary>
  /// Get live pentanomial computed from games completed so far.
  /// Safe to call during tournament (locks internally).
  /// Returns null if no games have completed yet.
  /// </summary>
  public int[] GetLivePentanomial()
  {
    lock (_statsLock)
    {
      return _gamePairsByOpening.Count == 0 ? null : ComputePentanomial();
    }
  }


  /// <summary>
  /// Stream completed game-pair results to the orchestrator via callback.
  /// Reads from _gamePairsByOpening (populated live during tournament).
  /// Each callback is awaited in order to prevent concurrent writes to the stream.
  /// </summary>
  private async Task StreamResultsAsync(GamePairCompletedHandler callback)
  {
    if (callback == null) return;

    // Snapshot the pairs under lock, then await each send outside the lock
    // so that stream writes are sequential (NetworkStream does not support concurrent writes).
    List<GamePairResult> pairs;
    lock (_statsLock)
    {
      pairs = _gamePairsByOpening
          .OrderBy(kv => kv.Key)
          .Select(kv => new GamePairResult
          {
            PerturbationId = _currentPerturbationId,
            OpeningIdx = kv.Key,
            R1 = kv.Value.r1,
            R2 = kv.Value.r2,
            CumulativeWDL = new[] { _wins, _draws, _losses }
          }).ToList();
    }

    foreach (var pair in pairs)
    {
      await callback(pair);
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


  /// <summary>
  /// Run a Ceres-vs-Ceres tournament between two networks.
  /// Both engines use the same search params and nodes per move.
  /// Results are from Engine 1 (NetA) perspective.
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

    // Configure search params (shared by both engines)
    var searchParams = new ParamsSearch();
    var selectParams = new ParamsSelect();
    searchParams.Execution.DualOverlappedIterators = false;
    searchParams.Execution.DualEvaluators = false;
    searchParams.Execution.NNBatchSizeAlignmentTarget = 0;
    ApplySearchParams(searchParams, selectParams, config.SearchParams ?? new());

    // Build net spec strings
    string net1Spec = (config.Net1Prefix ?? "") + config.Net1Path;
    if (!string.IsNullOrEmpty(config.Net1Options))
      net1Spec += "|" + config.Net1Options;

    string net2Spec = (config.Net2Prefix ?? "") + config.Net2Path;
    if (!string.IsNullOrEmpty(config.Net2Options))
      net2Spec += "|" + config.Net2Options;

    // Engine definitions — both are CeresMCGS on this worker's GPU
    string device = $"GPU:{_gpuId}#TensorRTNative";
    NNEvaluatorDef nn1Def = NNEvaluatorDefFactory.FromSpecification(net1Spec, device);
    NNEvaluatorDef nn2Def = NNEvaluatorDefFactory.FromSpecification(net2Spec, device);

    GameEngineDefCeresMCGS engineDef1 = new("NetA", nn1Def, searchParams, selectParams);
    GameEngineDefCeresMCGS engineDef2 = new("NetB", nn2Def, searchParams, selectParams);

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

    ct.Register(() =>
    {
      if (_currentDef != null)
        _currentDef.ShouldShutDown = true;
    });

    // Run tournament with live PerGamePairCallback.
    // GPU offset [0] because base GPU is already in NNEvaluatorDef (GPU:_gpuId#TensorRTNative).
    _liveGamesPlayed = 0;
    TournamentResultStats results = null;
    try
    {
      results = await Task.Run(() =>
      {
        int concurrency = Math.Max(1, config.Concurrency);
        int[] gpuIds = new[] { 0 };
        var mgr = new TournamentManager(def, concurrency, gpuIds);
        mgr.PerGamePairCallback = (gameInfo, gameReverseInfo) =>
        {
          // NetA is player index 0
          int r1 = CeresResult(gameInfo, 0);
          int r2 = CeresResult(gameReverseInfo, 0);
          lock (_statsLock)
          {
            if (r1 == 1) _wins++; else if (r1 == -1) _losses++; else _draws++;
            if (r2 == 1) _wins++; else if (r2 == -1) _losses++; else _draws++;
            _gamePairsByOpening[gameInfo.OpeningIndex] = (r1, r2);
            _liveGamesPlayed = _wins + _draws + _losses;
          }
        };
        return mgr.RunTournament(enableCancelVialCtrlC: false);
      }, ct);
    }
    catch (OperationCanceledException) { throw; }
    catch (Exception ex)
    {
      // Non-fatal: DumpTournamentSummary column width arithmetic bug.
      // Live W/D/L and _gamePairsByOpening are already populated via PerGamePairCallback.
      Console.Error.WriteLine($"[WorkerTournamentRunner] NetVsNet RunTournament warning (non-fatal): {ex.Message}");
    }

    // Stream results to orchestrator
    await StreamResultsAsync(onGamePairCompleted);

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
}
