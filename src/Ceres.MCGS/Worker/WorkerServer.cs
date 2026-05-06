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
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Chess.NNEvaluators.TensorRT;
using Ceres.Chess.UserSettings;

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

  // Reusable payload buffer to avoid LOH fragmentation from repeated ~29MB REFIT payloads.
  // Grows once to the max payload size and is reused for all subsequent commands.
  private byte[] _payloadBuffer;

  // Per-process durable journal of dispatched PLAY commands.  Persists
  // across worker process restarts via /tmp.  Used for idempotency on PLAY
  // (a cmd_id that's already COMPLETED returns the cached result instead of
  // replaying).  See WorkerJournal docstring for the fsync-before-send
  // correctness invariant.
  private readonly WorkerJournal _journal;

  // In-memory set of cmd_ids currently being executed by *this* process.
  // Wiped on process restart, unlike the journal.  Used to disambiguate the
  // journal STARTED-without-COMPLETED state:
  //   cmd_id ∈ _inFlight   → "still running right now, RESUME reports started"
  //   cmd_id has STARTED in journal but ∉ _inFlight → "abandoned by a prior
  //     process; on the next PLAY for this id we treat it as a fresh run".
  // Guarded by a plain HashSet under lock — contention is trivially low (a
  // few hundred adds/removes per iteration).
  private readonly HashSet<string> _inFlight = new();
  private readonly object _inFlightLock = new();


  public WorkerServer(int port, int gpuId, WorkerLocalConfig localConfig = null, string bindHost = "0.0.0.0")
  {
    _port = port;
    _gpuId = gpuId;
    _localConfig = localConfig ?? new WorkerLocalConfig();
    _bindHost = bindHost;

    // Build journal up-front (Replay happens in ctor) so the in-memory
    // cmd_id table is ready before the first client connection.  Replaying
    // takes O(file size) — at one entry per PLAY (~few hundred bytes each)
    // even 100k entries replays in <1s.
    _journal = new WorkerJournal(_gpuId);
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
      byte[] payload;
      if (length > 0 && cmd == WorkerCommand.Refit)
      {
        // Reuse payload buffer for REFIT to avoid LOH fragmentation.
        // REFIT payloads are ~29MB (32 weight tensors as FP16) and arrive every iteration.
        // Without reuse, .NET allocates new LOH segments that are never compacted,
        // causing unbounded host memory growth (~1.5GB/hour observed).
        if (_payloadBuffer == null || _payloadBuffer.Length < length)
        {
          _payloadBuffer = new byte[length];
        }
        await WorkerProtocol.ReadExactIntoAsync(stream, _payloadBuffer, length, ct);
        payload = _payloadBuffer;
      }
      else if (length > 0)
      {
        // Non-REFIT commands have small JSON payloads (< 1KB), safe to allocate fresh
        payload = await WorkerProtocol.ReadExactAsync(stream, length, ct);
      }
      else
      {
        payload = Array.Empty<byte>();
      }

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

        case WorkerCommand.NetVsNet:
          await HandleNetVsNetAsync(stream, payload, ct);
          break;

        case WorkerCommand.ListPlayedOffsets:
          await HandleListPlayedOffsetsAsync(stream, ct);
          break;

        case WorkerCommand.Resume:
          await HandleResumeAsync(stream, payload, ct);
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

      // Initialize tournament runner.  Pass the journal handle so the
      // inner game-pair loop can emit per-K-pair PROGRESS entries; the
      // server-side broadcast callback is wired up at PLAY time.
      _tournamentRunner = new WorkerTournamentRunner(
          ceresNetPath: netSpec,
          ceresJsonPath: config.CeresJsonPath,
          opponentExe: config.OpponentExe,
          opponentNodes: config.OpponentNodes,
          opponentThreads: config.OpponentThreads,
          engineNodes: config.EngineNodes,
          gpuId: _gpuId,
          bookPath: config.BookPath,
          searchParams: config.SearchParams,
          journal: _journal);

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

      // Serialize the refitted engine to a temp file so the tournament runner
      // (which loads its own NNEvaluator from the file path) uses the new weights.
      string serializePath = System.IO.Path.Combine(
          System.IO.Path.GetTempPath(), $"ceres_worker_gpu{_gpuId}_refitted.engine");
      var result = _refitter.Refit(perturbationId, weights, serializePath);

      // Update the tournament runner's engine path to the serialized file
      if (result.Status == "refitted" && _tournamentRunner != null)
      {
        _tournamentRunner.UpdateEnginePath(serializePath);
      }

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
    string cmdId = playConfig.CmdId;  // Nullable — older Python clients omit this.

    // Idempotency check.  If cmdId is provided AND the journal has a
    // terminal entry for it, replay the cached outcome instead of running
    // the tournament again.  Backwards compat: cmdId can be null (legacy
    // Python client) — in that case skip the journal entirely (no
    // idempotency, just log the absence).
    if (!string.IsNullOrEmpty(cmdId))
    {
      var prior = _journal.LookupCommand(cmdId);
      if (prior != null)
      {
        if (prior.Event == JournalEvent.COMPLETED.ToString())
        {
          // Idempotent replay: re-deserialize the cached TournamentResult and
          // return it.  Note we DO NOT call AppendStarted/AppendCompleted again
          // here — the journal already has COMPLETED, and we don't want to
          // duplicate fsyncs or grow the file on legitimate re-dispatch.
          // Release the semaphore manually since we're skipping the play path.
          Console.WriteLine(
            $"[Worker GPU:{_gpuId}] PLAY cmd_id={cmdId} already COMPLETED — replaying cached result");
          try
          {
            var cached = JsonSerializer.Deserialize<TournamentResult>(
                prior.Payload.GetRawText(),
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower });
            if (cached != null)
            {
              cached.CmdId = cmdId;
              await WorkerProtocol.SendResponseAsync(stream, cached, ct);
            }
            else
            {
              await WorkerProtocol.SendResponseAsync(stream, new
              {
                type = "error",
                cmd_id = cmdId,
                error = "Journal has COMPLETED entry but payload failed to deserialize"
              }, ct);
            }
          }
          finally
          {
            _opSemaphore.Release();
          }
          return;
        }
        else if (prior.Event == JournalEvent.STARTED.ToString())
        {
          // Disambiguate STARTED-without-COMPLETED via _inFlight.
          //   - If cmd_id is in _inFlight, another thread in *this* process is
          //     still running it.  Return an error telling the orchestrator
          //     to wait — the legitimate result will arrive on the original
          //     PLAY connection.  In normal operation the orchestrator's
          //     RESUME-before-PLAY discipline ensures we don't reach here, but
          //     we keep this branch defensive.
          //   - If cmd_id is NOT in _inFlight, the prior STARTED was written
          //     by a worker process that has since exited.  This is an
          //     "abandoned start" — fall through to the normal play path.
          //     The new AppendStarted (below) will overwrite the stale entry
          //     in the in-memory dict; on disk the journal has both, and the
          //     COMPLETED on the second attempt becomes the latest entry.
          bool currentlyRunning;
          lock (_inFlightLock)
          {
            currentlyRunning = _inFlight.Contains(cmdId);
          }

          if (currentlyRunning)
          {
            Console.WriteLine(
              $"[Worker GPU:{_gpuId}] PLAY cmd_id={cmdId} STARTED & in_progress in this process — refusing duplicate");
            try
            {
              await WorkerProtocol.SendResponseAsync(stream, new
              {
                type = "in_progress",
                cmd_id = cmdId,
                error = "Command currently running in this worker process; await results on the original connection."
              }, ct);
            }
            finally
            {
              _opSemaphore.Release();
            }
            return;
          }

          // Abandoned start: previous process began this cmd_id but never
          // completed it (crash, OOM, kill).  Treat as a fresh run.
          Console.WriteLine(
            $"[Worker GPU:{_gpuId}] PLAY cmd_id={cmdId} has STARTED in journal but not _inFlight " +
            "(abandoned by previous process) — treating as fresh run");
          // Fall through to the normal play path below.
        }
        // FAILED falls through: re-issuing a FAILED cmd_id under the same id
        // is allowed (the previous attempt is terminal, and the new append
        // will overwrite the in-memory entry).
      }
    }

    lock (_stateLock) { _currentPerturbationId = perturbationId; }
    _state = "playing";

    // Log cmd_id at INFO so we can verify plumbing end-to-end.  When
    // cmdId is null/empty the client is a legacy build (backwards compat).
    Console.WriteLine(
      string.IsNullOrEmpty(cmdId)
        ? $"[Worker GPU:{_gpuId}] PLAY '{perturbationId}' (no cmd_id — legacy client)"
        : $"[Worker GPU:{_gpuId}] PLAY '{perturbationId}' cmd_id={cmdId}");

    // Append STARTED entry BEFORE beginning execution.  fsync inside
    // AppendStarted ensures that on a worker crash + restart, the journal
    // will truthfully say "this cmd_id was begun but never completed".
    // Skip when cmdId is null (legacy client — no journal tracking possible).
    if (!string.IsNullOrEmpty(cmdId))
    {
      _journal.AppendStarted(cmdId, new
      {
        perturbation_id = perturbationId,
        num_game_pairs = playConfig.NumGamePairs,
        opening_offset = playConfig.OpeningOffset,
        concurrency = playConfig.Concurrency,
        opening_seed = playConfig.OpeningSeed
      });

      // Mark this cmd_id as currently running in this process.  The
      // pair to this Add is the Remove in the outermost finally block (after
      // _opSemaphore.Release).  HashSet under _inFlightLock — see field
      // declaration for the disambiguation rationale.
      lock (_inFlightLock)
      {
        _inFlight.Add(cmdId);
      }
    }

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
          onProgress: (progressCmdId, gamesPlayed, ofTotal, partialWdl) =>
              BroadcastProgressAsync(stream, progressCmdId, gamesPlayed, ofTotal, partialWdl, ct),
          ct: _tournamentCts.Token);

      _state = "idle";
      // Echo cmd_id back so the orchestrator can reconcile this result
      // against its dispatched_cmd_ids table.  Null when the client didn't
      // send one.
      result.CmdId = cmdId;

      // CRITICAL ORDERING (see WorkerJournal docstring):
      //   1. AppendCompleted (with fsync)  ← durability happens here
      //   2. SendResponseAsync             ← only after fsync confirms
      //
      // If we sent RESULTS first and then crashed before fsync, the
      // orchestrator would believe the work was done but on restart the
      // journal would say STARTED, and a re-dispatch of the same cmd_id
      // would re-play the work.  Sending after fsync guarantees: if
      // RESULTS arrived at the server, the journal definitely has COMPLETED.
      if (!string.IsNullOrEmpty(cmdId))
      {
        _journal.AppendCompleted(cmdId, result);
      }
      await WorkerProtocol.SendResponseAsync(stream, result, ct);

      // Check memory usage — auto-exit if RSS exceeds threshold.
      // The start script (screen with restart) will respawn the worker.
      // This is a safety net for the TRT engine memory leak where
      // NNEvaluator ref-counting prevents native CUDA memory from being freed.
      const long MAX_RSS_MB = 25_000; // 25 GB — headroom for iter-over-iter leak accumulation with adaptive chunking
      long rssMB = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
      if (rssMB > MAX_RSS_MB)
      {
        Console.WriteLine($"[Worker GPU:{_gpuId}] RSS {rssMB} MB exceeds {MAX_RSS_MB} MB limit — self-restarting");
        System.Environment.Exit(42); // Non-zero exit code signals intentional restart
      }
    }
    catch (OperationCanceledException)
    {
      _state = "idle";
      var (wins, draws, losses) = _tournamentRunner.GetCurrentWDL();
      int[] penta = _tournamentRunner.GetLivePentanomial();
      var stoppedResult = new TournamentResult
      {
        Type = "stopped",
        PerturbationId = perturbationId,
        Wins = wins,
        Draws = draws,
        Losses = losses,
        GamesPlayed = wins + draws + losses,
        Pentanomial = penta,
        CmdId = cmdId  // Echo even on stop/cancel so orchestrator can reconcile.
      };

      // A STOP cancellation isn't necessarily a failure (orchestrator
      // intentionally halted us) but the cmd_id was begun and not completed,
      // so journal it as FAILED with the partial WDL.  Keeps the same
      // fsync-before-send invariant as the success path.
      if (!string.IsNullOrEmpty(cmdId))
      {
        _journal.AppendFailed(cmdId, "operation_cancelled", stoppedResult);
      }
      await WorkerProtocol.SendResponseAsync(stream, stoppedResult, ct);
    }
    catch (Exception ex)
    {
      _state = "idle";
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] Tournament error: {ex}");
      var (wins, draws, losses) = _tournamentRunner?.GetCurrentWDL() ?? (0, 0, 0);
      int[] penta = _tournamentRunner?.GetLivePentanomial();

      // Persist the FAILED entry BEFORE sending the error response, for
      // the same fsync-before-send reason as the success path.
      if (!string.IsNullOrEmpty(cmdId))
      {
        _journal.AppendFailed(cmdId, ex.Message, new
        {
          perturbation_id = perturbationId,
          wins,
          draws,
          losses,
          games_played = wins + draws + losses,
          pentanomial = penta
        });
      }
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        perturbation_id = perturbationId,
        cmd_id = cmdId,  // Echo even on error path.
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
      // Clear the in-flight marker BEFORE releasing the operation
      // semaphore.  Order matters: a future PLAY for the same cmd_id will
      // queue on _opSemaphore and only proceed after Release; by then we want
      // _inFlight to already reflect that the previous run is no longer
      // executing in this process.
      if (!string.IsNullOrEmpty(cmdId))
      {
        lock (_inFlightLock)
        {
          _inFlight.Remove(cmdId);
        }
      }
      _opSemaphore.Release();
    }
  }


  /// <summary>
  /// Async PROGRESS push during PLAY.  Best-effort wire send: a dropped
  /// socket must NOT crash the runner's game loop, so all exceptions are
  /// swallowed and logged.  The orchestrator's read loop intercepts these
  /// before the next response, so they arrive in the natural read window of
  /// a still-blocked send_play call on the Python side.
  ///
  /// Called from WorkerTournamentRunner.PerGamePairCallback every K pairs.
  /// </summary>
  private async Task BroadcastProgressAsync(NetworkStream stream,
      string cmdId, int gamesPlayed, int ofTotal, int[] partialWdl,
      CancellationToken ct)
  {
    try
    {
      await WorkerProtocol.SendResponseAsync(stream, new ProgressMessage
      {
        Type = "progress",
        CmdId = cmdId,
        GamesPlayed = gamesPlayed,
        OfTotal = ofTotal,
        PartialWdl = partialWdl ?? Array.Empty<int>()
      }, ct);
    }
    catch (Exception ex)
    {
      // Socket dead, write failed, ct canceled — none should propagate.
      Console.Error.WriteLine(
        $"[Worker GPU:{_gpuId}] Failed to push PROGRESS for cmd_id={cmdId} " +
        $"({gamesPlayed}/{ofTotal}): {ex.Message}");
    }
  }


  /// <summary>
  /// RESUME — server-on-reconnect handshake.  The orchestrator sends
  /// a list of cmd_ids it dispatched (with its own view of expected state) and
  /// the worker replies with its journal slice for those cmd_ids plus any
  /// other recently-touched cmd_ids the server might not know about (e.g.
  /// completions that landed during the disconnect window).  See the design
  /// doc "Concept 3 — RESUME protocol" for the full contract.
  ///
  /// Correctness invariant: the orchestrator must not dispatch any new PLAY
  /// on a reconnected worker until RESUME_REPLY has been processed.  We rely
  /// on TCP message ordering plus the Python orchestrator's discipline (RESUME
  /// is sent on the same socket and awaits its reply before the next PLAY
  /// dispatch).  No server-side queueing flag is needed.
  /// </summary>
  private async Task HandleResumeAsync(NetworkStream stream, byte[] payload, CancellationToken ct)
  {
    try
    {
      var request = WorkerProtocol.ParseJson<ResumeRequest>(payload)
                    ?? new ResumeRequest();

      var queriedCmdIds = new HashSet<string>();
      if (request.ServerView != null)
      {
        foreach (var sv in request.ServerView)
        {
          if (!string.IsNullOrEmpty(sv?.CmdId)) queriedCmdIds.Add(sv.CmdId);
        }
      }

      // Snapshot the journal entries within the recency window.  This includes
      // everything within the last ~30 min PLUS gives us natural coverage of
      // cmd_ids the server didn't ask about (e.g. completed during a network gap).
      var recentEntries = _journal.SnapshotRecent();
      var seenCmdIds = new HashSet<string>(recentEntries.Count);
      var replyEntries = new List<ResumeReplyEntry>(recentEntries.Count);

      foreach (var entry in recentEntries)
      {
        if (string.IsNullOrEmpty(entry.CmdId)) continue;
        seenCmdIds.Add(entry.CmdId);

        var replyEntry = BuildResumeReplyEntry(entry);
        if (replyEntry != null) replyEntries.Add(replyEntry);
      }

      // For every cmd_id the server asked about that's NOT in our recency
      // window, look it up directly — it may live in the older portion of the
      // journal.  This handles the corner case of a long-running cmd_id whose
      // STARTED entry has aged out of the recency window but whose state the
      // server still cares about.
      foreach (var cmdId in queriedCmdIds)
      {
        if (seenCmdIds.Contains(cmdId)) continue;
        var entry = _journal.LookupCommand(cmdId);
        if (entry == null)
        {
          // Worker has no record of this cmd_id at all.  We deliberately do
          // NOT include it in the reply — absence is the orchestrator's signal
          // to re-dispatch (worker truly never saw it).
          continue;
        }
        var replyEntry = BuildResumeReplyEntry(entry);
        if (replyEntry != null) replyEntries.Add(replyEntry);
      }

      Console.WriteLine(
        $"[Worker GPU:{_gpuId}] RESUME: server asked about {queriedCmdIds.Count} cmd_ids, " +
        $"replying with {replyEntries.Count} entries (recency-window scan)");

      await WorkerProtocol.SendResponseAsync(stream, new ResumeReply
      {
        Type = "resume_reply",
        Status = "ok",
        Journal = replyEntries
      }, ct);
    }
    catch (Exception ex)
    {
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] RESUME error: {ex}");
      await WorkerProtocol.SendResponseAsync(stream, new ResumeReply
      {
        Type = "resume_reply",
        Status = "error",
        Journal = new List<ResumeReplyEntry>()
      }, ct);
    }
  }


  /// <summary>
  /// Helper: convert one journal entry into a ResumeReplyEntry suitable
  /// for the wire.  The "started" → in_progress mapping is decided here based
  /// on whether the cmd_id is currently in _inFlight (this process is still
  /// executing it) vs an abandoned start from a prior process.  Either way the
  /// outgoing State is "started" — the orchestrator's reconcile logic uses
  /// that string verbatim to mean "leave it alone, expect a future RESULTS"
  /// when it's a current run, or to mean "treat as failed and requeue" when
  /// it's an abandoned start.  We surface that distinction by appending a
  /// suffix in logs only; the wire shape stays clean.
  /// </summary>
  private ResumeReplyEntry BuildResumeReplyEntry(JournalEntry entry)
  {
    string evt = entry.Event ?? "";
    if (evt == JournalEvent.COMPLETED.ToString())
    {
      TournamentResult cached = null;
      try
      {
        cached = JsonSerializer.Deserialize<TournamentResult>(
            entry.Payload.GetRawText(),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower });
        if (cached != null) cached.CmdId = entry.CmdId;
      }
      catch (Exception ex)
      {
        Console.Error.WriteLine(
          $"[Worker GPU:{_gpuId}] RESUME: failed to deserialize cached result for {entry.CmdId}: {ex.Message}");
      }

      return new ResumeReplyEntry
      {
        CmdId = entry.CmdId,
        State = "completed",
        Results = cached
      };
    }
    if (evt == JournalEvent.STARTED.ToString())
    {
      bool currentlyRunning;
      lock (_inFlightLock)
      {
        currentlyRunning = _inFlight.Contains(entry.CmdId);
      }
      // The wire state is "started" in both sub-cases — the orchestrator
      // doesn't care whether we're still running it or not, only that no
      // COMPLETED/FAILED has been written.  Logging notes the distinction.
      if (!currentlyRunning)
      {
        Console.WriteLine(
          $"[Worker GPU:{_gpuId}] RESUME: cmd_id={entry.CmdId} STARTED but not _inFlight (abandoned)");
      }
      return new ResumeReplyEntry
      {
        CmdId = entry.CmdId,
        State = "started",
        Results = null
      };
    }
    if (evt == JournalEvent.FAILED.ToString())
    {
      return new ResumeReplyEntry
      {
        CmdId = entry.CmdId,
        State = "failed",
        Results = null
      };
    }
    // PROGRESS or unknown event types — skip in the RESUME reply.  The journal's
    // most-recent-event-per-cmd_id contract means a PROGRESS-only entry
    // shouldn't normally be the last word, but if it is, treat as "started".
    return new ResumeReplyEntry
    {
      CmdId = entry.CmdId,
      State = "started",
      Results = null
    };
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
  /// LIST_PLAYED_OFFSETS: Return a snapshot of every completed game pair in
  /// the CURRENT tournament. Used by the orchestrator's reconnect-recovery
  /// path to replay game_pair events that were emitted while its stream was
  /// dropped — avoiding duplicate play by rescue workers on offsets the
  /// orphaned PLAY had already finished.
  ///
  /// Safe in any state: returns an empty list (state="idle"/"uninitialized")
  /// when nothing is running. Does NOT affect tournament state.
  /// </summary>
  private async Task HandleListPlayedOffsetsAsync(NetworkStream stream, CancellationToken ct)
  {
    string perturbationId;
    lock (_stateLock) { perturbationId = _currentPerturbationId; }

    var entries = new List<PlayedOffsetEntry>();
    if (_tournamentRunner != null)
    {
      foreach (var (openingIdx, r1, r2) in _tournamentRunner.GetPlayedOffsetsSnapshot())
      {
        entries.Add(new PlayedOffsetEntry { OpeningIdx = openingIdx, R1 = r1, R2 = r2 });
      }
    }

    await WorkerProtocol.SendResponseAsync(stream, new PlayedOffsetsResult
    {
      PerturbationId = perturbationId,
      State = _state,
      Offsets = entries
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
    lock (_stateLock) { _currentPerturbationId = "netvsnet"; }

    _tournamentCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

    try
    {
      var config = WorkerProtocol.ParseJson<NetVsNetConfig>(payload);
      Console.WriteLine($"[Worker GPU:{_gpuId}] NETVSNET: {config.Net1Path} vs {config.Net2Path}, " +
                        $"{config.NumGamePairs} pairs @ {config.NodesPerMove} nodes");

      // Load Ceres settings for tablebase dir etc. (needed even without prior INIT)
      if (!string.IsNullOrEmpty(_localConfig.CeresJsonPath))
        CeresUserSettingsManager.LoadFromFile(_localConfig.CeresJsonPath);

      // Create a temporary tournament runner for this command
      var runner = new WorkerTournamentRunner(
          ceresNetPath: "",
          ceresJsonPath: _localConfig.CeresJsonPath,
          opponentExe: "",
          opponentNodes: 0,
          opponentThreads: 0,
          engineNodes: config.NodesPerMove,
          gpuId: _gpuId,
          bookPath: _localConfig.BookPath,
          searchParams: config.SearchParams ?? new Dictionary<string, double>());

      var result = await runner.RunNetVsNetAsync(
          config,
          onGamePairCompleted: async (gamePairResult) =>
          {
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
      var (wins, draws, losses) = _tournamentRunner?.GetCurrentWDL() ?? (0, 0, 0);
      int[] penta = _tournamentRunner?.GetLivePentanomial();
      await WorkerProtocol.SendResponseAsync(stream, new TournamentResult
      {
        Type = "stopped",
        PerturbationId = "netvsnet",
        Wins = wins,
        Draws = draws,
        Losses = losses,
        GamesPlayed = wins + draws + losses,
        Pentanomial = penta
      }, ct);
    }
    catch (Exception ex)
    {
      _state = previousState;
      Console.Error.WriteLine($"[Worker GPU:{_gpuId}] NETVSNET error: {ex.Message}");
      await WorkerProtocol.SendResponseAsync(stream, new
      {
        type = "error",
        error = ex.Message
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
