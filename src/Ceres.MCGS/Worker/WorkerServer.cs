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
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Chess.NNEvaluators.TensorRT;

#endregion

namespace Ceres.MCGS.Worker;

/// <summary>
/// TCP server for a persistent Ceres worker process.
///
/// Lifecycle:
/// 1. Worker starts, loads a refittable TRT engine + opening book on a specific GPU
/// 2. Listens for TCP connections from the Python orchestrator
/// 3. Handles commands: INIT, REFIT, PLAY, STOP, STATUS, SHUTDOWN
/// 4. Between tournaments, REFIT replaces engine weights in-place (no reload)
/// 5. PLAY runs games and streams per-game-pair results back
/// 6. STOP halts the current tournament gracefully
///
/// Only one client connection at a time is expected (the orchestrator).
/// </summary>
public class WorkerServer
{
  private readonly int _port;
  private readonly int _gpuId;
  private readonly string _bindHost;
  private readonly WorkerLocalConfig _localConfig;
  private readonly CancellationTokenSource _shutdownCts = new();

  // Engine state
  private TensorRTEngine[] _engines;
  private WorkerRefitter _refitter;
  private WorkerTournamentRunner _tournamentRunner;
  private CancellationTokenSource _tournamentCts;

  // Worker state — protected by _stateLock for reads/writes from concurrent connections
  private readonly object _stateLock = new();
  private volatile string _state = "uninitialized";  // uninitialized, idle, playing, refitting
  private string _currentPerturbationId;

  // Semaphore: only one REFIT or PLAY allowed at a time
  private readonly SemaphoreSlim _opSemaphore = new(1, 1);


  public WorkerServer(int port, int gpuId, WorkerLocalConfig localConfig = null, string bindHost = "0.0.0.0")
  {
    _port = port;
    _gpuId = gpuId;
    _localConfig = localConfig ?? new WorkerLocalConfig();
    _bindHost = bindHost;
  }


  /// <summary>
  /// Start the worker server. Blocks until SHUTDOWN received or cancelled.
  /// </summary>
  public async Task RunAsync()
  {
    IPAddress bindAddr = _bindHost == "0.0.0.0" ? IPAddress.Any : IPAddress.Parse(_bindHost);
    TcpListener listener = new(bindAddr, _port);
    listener.Start();
    Console.WriteLine($"[Worker GPU:{_gpuId}] Listening on port {_port}");

    try
    {
      while (!_shutdownCts.Token.IsCancellationRequested)
      {
        TcpClient client;
        try
        {
          client = await listener.AcceptTcpClientAsync(_shutdownCts.Token);
        }
        catch (OperationCanceledException)
        {
          break;
        }

        Console.WriteLine($"[Worker GPU:{_gpuId}] Client connected from {client.Client.RemoteEndPoint}");

        // Handle each connection concurrently so STOP/STATUS can arrive on a new connection
        // while PLAY is streaming results on the primary connection.
        _ = Task.Run(async () =>
        {
          try
          {
            await HandleClientAsync(client, _shutdownCts.Token);
          }
          catch (Exception ex)
          {
            Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Client error: {ex.Message}");
          }
          finally
          {
            client.Close();
            Console.WriteLine($"[Worker GPU:{_gpuId}] Client disconnected");
          }
        });
      }
    }
    finally
    {
      listener.Stop();
      Console.WriteLine($"[Worker GPU:{_gpuId}] Server stopped");
    }
  }


  /// <summary>
  /// Handle a single client connection — reads commands in a loop.
  /// </summary>
  private async Task HandleClientAsync(TcpClient client, CancellationToken ct)
  {
    using var stream = client.GetStream();

    while (!ct.IsCancellationRequested)
    {
      var header = await WorkerProtocol.ReadCommandHeaderAsync(stream, ct);
      if (header == null) break;  // Client disconnected

      var (cmd, length) = header.Value;
      byte[] payload = length > 0
          ? await WorkerProtocol.ReadExactAsync(stream, length, ct)
          : Array.Empty<byte>();

      Console.WriteLine($"[Worker GPU:{_gpuId}] Command: {cmd} ({length} bytes)");

      switch (cmd)
      {
        case WorkerCommand.Init:
          await HandleInitAsync(stream, payload, ct);
          break;

        case WorkerCommand.Refit:
          await HandleRefitAsync(stream, payload, ct);
          break;

        case WorkerCommand.Play:
          await HandlePlayAsync(stream, payload, ct);
          break;

        case WorkerCommand.Stop:
          await HandleStopAsync(stream, ct);
          break;

        case WorkerCommand.Status:
          await HandleStatusAsync(stream, ct);
          break;

        case WorkerCommand.ProbeDeps:
          await HandleProbeDepAsync(stream, payload, ct);
          break;

        case WorkerCommand.Serialize:
          await HandleSerializeAsync(stream, payload, ct);
          break;

        case WorkerCommand.Shutdown:
          await HandleShutdownAsync(stream, ct);
          return;
      }
    }
  }


  /// <summary>
  /// INIT: Load engine and configure tournament parameters.
  /// </summary>
  private async Task HandleInitAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    try
    {
      var config = WorkerProtocol.ParseJson<InitConfig>(payload);

      // Fill server-local paths from local config if orchestrator didn't provide them
      if (string.IsNullOrEmpty(config.BookPath))      config.BookPath      = _localConfig.BookPath;
      if (string.IsNullOrEmpty(config.OpponentExe))   config.OpponentExe   = _localConfig.OpponentExe;
      if (string.IsNullOrEmpty(config.CeresJsonPath)) config.CeresJsonPath = _localConfig.CeresJsonPath;

      Console.WriteLine($"[Worker GPU:{_gpuId}] Loading engine from {config.EnginePath}");
      var sw = Stopwatch.StartNew();

      // Initialize TRT runtime if not already done (required before any engine load).
      // In the normal Ceres flow this happens inside NNEvaluatorTensorRT; in worker
      // mode we call it explicitly here.
      bool firstInit = TensorRTEngine.EnsureInitialized();
      Console.WriteLine($"[Worker GPU:{_gpuId}] TRT runtime {(firstInit ? "initialized" : "already initialized")}");

      // Load multi-profile engine on this GPU
      int[] batchSizes = config.BatchSizes ?? new[] { 1, 2, 4, 8 };
      _engines = TensorRTEngine.LoadMultiProfileEngineFile(
          config.EnginePath,
          batchSizes,
          useCudaGraphs: config.UseCudaGraphs,
          useSpinWait: true,
          deviceId: _gpuId);

      sw.Stop();
      Console.WriteLine($"[Worker GPU:{_gpuId}] Engine loaded ({_engines.Length} profiles) in {sw.ElapsedMilliseconds}ms");

      // Initialize refitter
      _refitter = new WorkerRefitter(_engines);

      // Build the network spec string for Ceres tournament engine
      // e.g. "ONNX_TRT:/path/to/engine.engine|cudagraphs=true;V1TEMP=0.6989"
      string netSpec = config.NetPrefix + config.EnginePath;
      if (!string.IsNullOrEmpty(config.NetOptions))
      {
        netSpec += "|" + config.NetOptions;
      }

      // Initialize tournament runner
      _tournamentRunner = new WorkerTournamentRunner(
          ceresNetPath: netSpec,
          ceresJsonPath: config.CeresJsonPath,
          opponentExe: config.OpponentExe,
          opponentNodes: config.OpponentNodes,
          opponentThreads: config.OpponentThreads,
          engineNodes: config.EngineNodes,
          gpuId: _gpuId,
          bookPath: config.BookPath,
          searchParams: config.SearchParams);

      _state = "idle";

      await WorkerProtocol.SendResponseAsync(stream, new
      {
        status = "ready",
        gpu_id = _gpuId,
        profiles = batchSizes.Length,
        load_time_ms = sw.ElapsedMilliseconds
      }, ct);
    }
    catch (Exception ex)
    {
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Init failed: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        status = "error",
        error = ex.Message
      }, ct);
    }
  }


  /// <summary>
  /// REFIT: Replace engine weights in-place.
  /// </summary>
  private async Task HandleRefitAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    if (_refitter == null)
    {
      await WorkerProtocol.SendResponseAsync(stream, new RefitResult
      {
        Status = "error",
        Error = "Worker not initialized — send INIT first"
      }, ct);
      return;
    }

    // Acquire semaphore — prevents REFIT while PLAY is running (non-blocking)
    if (!await _opSemaphore.WaitAsync(0, ct))
    {
      await WorkerProtocol.SendResponseAsync(stream, new RefitResult
      {
        Status = "error",
        Error = "Worker busy — PLAY in progress. Send STOP first."
      }, ct);
      return;
    }

    _state = "refitting";

    try
    {
      var (perturbationId, weights) = WorkerProtocol.ParseRefitPayload(payload);
      var result = _refitter.Refit(perturbationId, weights);
      _state = "idle";
      await WorkerProtocol.SendResponseAsync(stream, result, ct);
    }
    catch (Exception ex)
    {
      _state = "idle";
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Refit error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new RefitResult
      {
        Status = "error",
        Error = ex.Message
      }, ct);
    }
    finally
    {
      _opSemaphore.Release();
    }
  }


  /// <summary>
  /// PLAY: Start a tournament with the current engine weights.
  /// Streams game-pair results, then sends final tournament result.
  /// </summary>
  private async Task HandlePlayAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    if (_tournamentRunner == null)
    {
      await WorkerProtocol.SendResponseAsync(stream, new TournamentResult
      {
        Type = "error",
        PerturbationId = "unknown"
      }, ct);
      return;
    }

    // Acquire semaphore — only one REFIT or PLAY at a time (non-blocking check)
    if (!await _opSemaphore.WaitAsync(0, ct))
    {
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        error = "Worker busy — REFIT or PLAY already in progress"
      }, ct);
      return;
    }

    var playConfig = WorkerProtocol.ParseJson<PlayConfig>(payload);
    string perturbationId = playConfig.PerturbationId;
    lock (_stateLock) { _currentPerturbationId = perturbationId; }
    _state = "playing";

    _tournamentCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

    try
    {
      // Run tournament — game pairs streamed to orchestrator after tournament completes.
      // Live W/D/L and pentanomial are tracked inside WorkerTournamentRunner via
      // TournamentManager.PerGamePairCallback (queryable via STATUS during play).
      var result = await _tournamentRunner.RunAsync(
          playConfig,
          onGamePairCompleted: async (gamePairResult) =>
          {
            // Stream each completed game pair back to the orchestrator.
            try
            {
              await WorkerProtocol.SendResponseAsync(stream, gamePairResult, ct);
            }
            catch (Exception ex)
            {
              Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Failed to stream game pair: {ex.Message}");
            }
          },
          ct: _tournamentCts.Token);

      _state = "idle";
      await WorkerProtocol.SendResponseAsync(stream, result, ct);
    }
    catch (OperationCanceledException)
    {
      _state = "idle";
      var (wins, draws, losses) = _tournamentRunner.GetCurrentWDL();
      int[] penta = _tournamentRunner.GetLivePentanomial();
      await WorkerProtocol.SendResponseAsync(stream, new TournamentResult
      {
        Type = "stopped",
        PerturbationId = perturbationId,
        Wins = wins,
        Draws = draws,
        Losses = losses,
        GamesPlayed = wins + draws + losses,
        Pentanomial = penta
      }, ct);
    }
    catch (Exception ex)
    {
      _state = "idle";
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Tournament error: {ex}");
      var (wins, draws, losses) = _tournamentRunner?.GetCurrentWDL() ?? (0, 0, 0);
      int[] penta = _tournamentRunner?.GetLivePentanomial();
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        perturbation_id = perturbationId,
        error = ex.Message,
        wins,
        draws,
        losses,
        games_played = wins + draws + losses,
        pentanomial = penta
      }, ct);
    }
    finally
    {
      _opSemaphore.Release();
    }
  }


  /// <summary>
  /// STOP: Stop the current tournament gracefully.
  /// </summary>
  private async Task HandleStopAsync(NetworkStream stream, CancellationToken ct)
  {
    if (_state != "playing")
    {
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "not_playing",
        state = _state
      }, ct);
      return;
    }

    Console.WriteLine($"[Worker GPU:{_gpuId}] Stopping tournament for '{_currentPerturbationId}'");
    _tournamentRunner?.Stop();
    _tournamentCts?.Cancel();

    await WorkerProtocol.SendResponseAsync(stream, new
    {
      type = "stopping",
      perturbation_id = _currentPerturbationId
    }, ct);
  }


  /// <summary>
  /// STATUS: Return current worker state.
  /// </summary>
  private async Task HandleStatusAsync(NetworkStream stream, CancellationToken ct)
  {
    string perturbationId;
    lock (_stateLock) { perturbationId = _currentPerturbationId; }

    int gamesPlayed = _tournamentRunner?.GetLiveGamesPlayed() ?? 0;
    var (w, d, l) = _tournamentRunner != null ? _tournamentRunner.GetCurrentWDL() : (0, 0, 0);
    int[] penta = _tournamentRunner?.GetLivePentanomial();

    await WorkerProtocol.SendResponseAsync(stream, new WorkerStatus
    {
      State = _state,
      PerturbationId = perturbationId,
      GamesPlayed = gamesPlayed,
      WDL = new[] { w, d, l },
      GpuId = _gpuId,
      Pentanomial = penta
    }, ct);
  }


  /// <summary>
  /// PROBE_DEPS: Discover fused TRT dependency names for the supplied weight names.
  /// The engine state is NOT modified — this is a read-only query.
  /// </summary>
  private async Task HandleProbeDepAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    if (_refitter == null)
    {
      await WorkerProtocol.SendResponseAsync(stream, new ProbeDepsResult
      {
        Status = "error",
        Error = "Worker not initialized — send INIT first"
      }, ct);
      return;
    }

    try
    {
      var req = WorkerProtocol.ParseJson<ProbeDepsRequest>(payload);
      List<string> fusedDeps = _refitter.GetFusedDeps(req.WeightNames);

      Console.WriteLine($"[Worker GPU:{_gpuId}] ProbeDeps: {req.WeightNames.Count} user weights → {fusedDeps.Count} fused deps");

      await WorkerProtocol.SendResponseAsync(stream, new ProbeDepsResult
      {
        Status = "ok",
        FusedDeps = fusedDeps,
        UserWeights = req.WeightNames.Count
      }, ct);
    }
    catch (Exception ex)
    {
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] ProbeDeps error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new ProbeDepsResult
      {
        Status = "error",
        Error = ex.Message
      }, ct);
    }
  }


  /// <summary>
  /// SERIALIZE: Save the current engine weights to a file on the worker host.
  /// Useful for validation: refit then serialize, evaluate the resulting engine file with Ceres.
  /// </summary>
  private async Task HandleSerializeAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    if (_engines == null)
    {
      await WorkerProtocol.SendResponseAsync(stream, new SerializeResult
      {
        Status = "error",
        Error = "Worker not initialized — send INIT first"
      }, ct);
      return;
    }

    try
    {
      var req = WorkerProtocol.ParseJson<SerializeRequest>(payload);
      if (string.IsNullOrEmpty(req.OutputPath))
        throw new ArgumentException("output_path must be non-empty");

      // Ensure destination directory exists
      string dir = Path.GetDirectoryName(req.OutputPath);
      if (!string.IsNullOrEmpty(dir))
        Directory.CreateDirectory(dir);

      // Serialize via the first engine handle (all profiles share the same ICudaEngine)
      _engines[0].SaveEngine(req.OutputPath);
      long sizeBytes = new FileInfo(req.OutputPath).Length;

      Console.WriteLine($"[Worker GPU:{_gpuId}] Serialized engine to {req.OutputPath} ({sizeBytes / 1024 / 1024.0:F1} MB)");

      await WorkerProtocol.SendResponseAsync(stream, new SerializeResult
      {
        Status = "ok",
        OutputPath = req.OutputPath,
        SizeBytes = sizeBytes
      }, ct);
    }
    catch (Exception ex)
    {
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Serialize error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new SerializeResult
      {
        Status = "error",
        Error = ex.Message
      }, ct);
    }
  }


  /// <summary>
  /// SHUTDOWN: Clean exit.
  /// </summary>
  private async Task HandleShutdownAsync(NetworkStream stream, CancellationToken ct)
  {
    Console.WriteLine($"[Worker GPU:{_gpuId}] Shutting down...");

    // Stop any running tournament
    _tournamentRunner?.Stop();
    _tournamentCts?.Cancel();

    // Dispose engines
    if (_engines != null)
    {
      foreach (var engine in _engines)
      {
        engine?.Dispose();
      }
    }

    await WorkerProtocol.SendResponseAsync(stream, new
    {
      status = "shutting_down"
    }, ct);

    _shutdownCts.Cancel();
  }


  /// <summary>
  /// Entry point for launching a worker from the command line.
  ///
  /// Usage:
  ///   Ceres.MCGS --worker --worker-config /path/to/worker_config_gpu0.json
  ///   Ceres.MCGS --worker --gpu 0 --port 5100          (legacy, no local config)
  ///   Ceres.MCGS --worker --worker-config ... --gpu 1  (config + GPU override)
  ///
  /// CLI overrides (--gpu, --port, --host) take precedence over config file values.
  /// </summary>
  public static async Task LaunchWorkerAsync(
      int gpuId, int port, WorkerLocalConfig localConfig = null, string bindHost = "0.0.0.0")
  {
    Console.WriteLine($"=== Ceres Worker Mode ===");
    Console.WriteLine($"GPU: {gpuId}, Port: {port}, Bind: {bindHost}");
    if (localConfig != null)
    {
      Console.WriteLine($"Local config: SF={localConfig.OpponentExe}, Book={localConfig.BookPath}");
      Console.WriteLine($"              CeresJson={localConfig.CeresJsonPath}");
    }
    Console.WriteLine($"Waiting for INIT command from orchestrator...");
    Console.WriteLine();

    var server = new WorkerServer(port, gpuId, localConfig, bindHost);
    await server.RunAsync();
  }
}
