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
  private readonly WorkerLocalConfig _localConfig;
  private readonly CancellationTokenSource _shutdownCts = new();

  // Engine state
  private TensorRTEngine[] _engines;
  private WorkerRefitter _refitter;
  private WorkerTournamentRunner _tournamentRunner;
  private CancellationTokenSource _tournamentCts;
  private Task<TournamentResult> _tournamentTask;

  // Worker state
  private string _state = "uninitialized";  // uninitialized, idle, playing, refitting
  private string _currentPerturbationId;
  private int _gamesPlayed;
  private int[] _currentWDL = new[] { 0, 0, 0 };


  public WorkerServer(int port, int gpuId, WorkerLocalConfig localConfig = null, string bindHost = "0.0.0.0")
  {
    _port = port;
    _gpuId = gpuId;
    _localConfig = localConfig ?? new WorkerLocalConfig();
  }


  /// <summary>
  /// Start the worker server. Blocks until SHUTDOWN received or cancelled.
  /// </summary>
  public async Task RunAsync()
  {
    TcpListener listener = new(IPAddress.Any, _port);
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

        case WorkerCommand.NetVsNet:
          await HandleNetVsNetAsync(stream, payload, ct);
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

      Console.WriteLine($"[Worker GPU:{_gpuId}] Loading engine from {config.EnginePath}");
      var sw = Stopwatch.StartNew();

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

      // Apply local config fallbacks for server-specific paths
      if (string.IsNullOrEmpty(config.BookPath))      config.BookPath      = _localConfig.BookPath;
      if (string.IsNullOrEmpty(config.OpponentExe))   config.OpponentExe   = _localConfig.OpponentExe;
      if (string.IsNullOrEmpty(config.CeresJsonPath)) config.CeresJsonPath = _localConfig.CeresJsonPath;

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

    if (_state == "playing")
    {
      await WorkerProtocol.SendResponseAsync(stream, new RefitResult
      {
        Status = "error",
        Error = "Cannot refit while tournament is running — send STOP first"
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

    if (_state == "playing")
    {
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        error = "Tournament already running"
      }, ct);
      return;
    }

    var playConfig = WorkerProtocol.ParseJson<PlayConfig>(payload);
    _currentPerturbationId = playConfig.PerturbationId;
    _gamesPlayed = 0;
    _currentWDL = new[] { 0, 0, 0 };
    _state = "playing";

    _tournamentCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

    try
    {
      // Run tournament and stream results
      var result = await _tournamentRunner.RunAsync(
          playConfig,
          onGamePairCompleted: async (gamePairResult) =>
          {
            _gamesPlayed = gamePairResult.CumulativeWDL[0] +
                           gamePairResult.CumulativeWDL[1] +
                           gamePairResult.CumulativeWDL[2];
            _currentWDL = gamePairResult.CumulativeWDL;

            // Stream game pair result to orchestrator
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

      // Send final result
      await WorkerProtocol.SendResponseAsync(stream, result, ct);
    }
    catch (OperationCanceledException)
    {
      _state = "idle";
      // Tournament was stopped — send partial results
      var (wins, draws, losses) = _tournamentRunner.GetCurrentWDL();
      await WorkerProtocol.SendResponseAsync(stream, new TournamentResult
      {
        Type = "stopped",
        PerturbationId = _currentPerturbationId,
        Wins = wins,
        Draws = draws,
        Losses = losses,
        GamesPlayed = wins + draws + losses
      }, ct);
    }
    catch (Exception ex)
    {
      _state = "idle";
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Tournament error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        perturbation_id = _currentPerturbationId,
        error = ex.Message,
        games_completed = _gamesPlayed
      }, ct);
    }
  }


  /// <summary>
  /// NETVSNET: Run a Ceres-vs-Ceres tournament between two networks.
  /// Self-contained — does not require prior INIT. Uses the worker's GPU and local config.
  /// </summary>
  private async Task HandleNetVsNetAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    if (_state == "playing")
    {
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        error = "Cannot start NETVSNET while tournament is running — send STOP first"
      }, ct);
      return;
    }

    var previousState = _state;
    _state = "playing";
    _gamesPlayed = 0;
    _currentWDL = new[] { 0, 0, 0 };
    _currentPerturbationId = "netvsnet";

    _tournamentCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

    try
    {
      var config = WorkerProtocol.ParseJson<NetVsNetConfig>(payload);
      Console.WriteLine($"[Worker GPU:{_gpuId}] NETVSNET: {config.Net1Path} vs {config.Net2Path}, " +
                        $"{config.NumGamePairs} pairs @ {config.NodesPerMove} nodes");

      // Create a temporary tournament runner for this command.
      // The runner uses the worker's stored book path and Ceres settings (if loaded via prior INIT).
      // For net-vs-net, ceresNetPath/opponentExe are not used — RunNetVsNetAsync builds its own engines.
      var runner = new WorkerTournamentRunner(
          ceresNetPath: "",  // Not used for net-vs-net
          ceresJsonPath: "", // Not used for net-vs-net (settings already loaded or will use defaults)
          opponentExe: "",
          opponentNodes: 0,
          opponentThreads: 0,
          engineNodes: config.NodesPerMove,
          gpuId: _gpuId,
          bookPath: "",
          searchParams: config.SearchParams ?? new Dictionary<string, double>());

      var result = await runner.RunNetVsNetAsync(
          config,
          onGamePairCompleted: async (gamePairResult) =>
          {
            _gamesPlayed = gamePairResult.CumulativeWDL[0] +
                           gamePairResult.CumulativeWDL[1] +
                           gamePairResult.CumulativeWDL[2];
            _currentWDL = gamePairResult.CumulativeWDL;

            try
            {
              await WorkerProtocol.SendResponseAsync(stream, gamePairResult, ct);
            }
            catch (Exception ex)
            {
              Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Failed to stream NETVSNET game pair: {ex.Message}");
            }
          },
          ct: _tournamentCts.Token);

      _state = previousState;
      await WorkerProtocol.SendResponseAsync(stream, result, ct);
    }
    catch (OperationCanceledException)
    {
      _state = previousState;
      await WorkerProtocol.SendResponseAsync(stream, new TournamentResult
      {
        Type = "stopped",
        PerturbationId = "netvsnet",
        Wins = _currentWDL[0],
        Draws = _currentWDL[1],
        Losses = _currentWDL[2],
        GamesPlayed = _currentWDL[0] + _currentWDL[1] + _currentWDL[2]
      }, ct);
    }
    catch (Exception ex)
    {
      _state = previousState;
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] NETVSNET error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        error = ex.Message,
        games_completed = _gamesPlayed
      }, ct);
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
    await WorkerProtocol.SendResponseAsync(stream, new WorkerStatus
    {
      State = _state,
      PerturbationId = _currentPerturbationId,
      GamesPlayed = _gamesPlayed,
      WDL = _currentWDL,
      GpuId = _gpuId
    }, ct);
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
  /// Usage: Ceres.MCGS --worker --gpu 0 --port 5100
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
