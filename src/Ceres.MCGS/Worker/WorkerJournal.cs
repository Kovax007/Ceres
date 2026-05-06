#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

// TODO(follow-up): the journal currently grows unbounded over the lifetime
// of the worker process.  In a long tune (10k+ iterations) this would
// eventually consume non-trivial /tmp space.  Future work: rotate at
// iteration boundaries (rename to .iterN.journal, archive), truncate after
// N entries, or compact on Replay() by dropping entries that have been
// COMPLETED/FAILED for >M iterations.  See state_machine_sync_design.md
// "Risks / things to test → Journal size growth".

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

#endregion

namespace Ceres.MCGS.Worker;


/// <summary>
/// Event types persisted to the per-worker journal.  STARTED/COMPLETED/FAILED
/// are durable (fsynced).  PROGRESS is best-effort by default; the runner
/// only fsyncs the final pre-COMPLETED progress tick.
/// </summary>
public enum JournalEvent
{
  STARTED,
  PROGRESS,
  COMPLETED,
  FAILED
}


/// <summary>
/// One on-disk journal entry.  Serialized as JSON, one entry per line (JSONL).
/// The Payload is whatever the caller passed in; we don't constrain its shape
/// because different events carry different fields (STARTED has the PlayConfig,
/// COMPLETED has the TournamentResult, FAILED has a reason + partial WDL).
/// </summary>
public class JournalEntry
{
  public double Ts { get; set; }
  public string CmdId { get; set; }
  public string Event { get; set; }
  public JsonElement Payload { get; set; }
}


/// <summary>
/// Per-worker append-only journal of dispatched work.  See
/// state_machine_sync_design.md "Concept 2 — Worker-side journal" for the
/// full contract.  Events written: STARTED, PROGRESS, COMPLETED, FAILED.
///
/// File path: /tmp/ceres_worker_gpu{N}.journal (one journal per worker process).
/// Format:    JSONL — one {ts, cmd_id, event, payload} object per line.
///
/// Concurrency model: a single lock guards every Append + Flush + fsync.
/// The in-memory dict is rebuilt on construction via Replay() and is updated
/// in-place by every successful Append.  Reads from the dict take the same
/// lock so callers always see the post-fsync state.
///
/// Critical correctness invariant (see design doc, "Race: COMPLETED journal
/// entry vs RESULTS network message"): callers MUST fsync the COMPLETED entry
/// BEFORE sending RESULTS over the wire.  Each Append* method here does the
/// fsync inline before returning, so the invariant reduces to "call
/// AppendCompleted, then SendResponseAsync" in the right order at the call
/// site.  See WorkerServer.HandlePlayAsync for the ordering.
/// </summary>
public class WorkerJournal
{
  private readonly int _gpuId;
  private readonly string _path;
  private readonly object _lock = new();

  // cmd_id → latest event for that cmd_id.  "Latest" by file order on disk
  // (which equals real time order because every Append takes _lock and fsyncs
  // before releasing it, so two appends are strictly serialized).
  private readonly Dictionary<string, JournalEntry> _byCmdId = new();

  private static readonly JsonSerializerOptions JsonOpts = new()
  {
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    WriteIndented = false
  };


  public WorkerJournal(int gpuId)
  {
    _gpuId = gpuId;
    _path = $"/tmp/ceres_worker_gpu{_gpuId}.journal";

    Replay();
    Console.WriteLine(
      $"[Worker GPU:{_gpuId}] Journal at {_path} — replayed {_byCmdId.Count} cmd_id entries");
  }


  /// <summary>Path to the on-disk journal file.</summary>
  public string Path => _path;


  /// <summary>
  /// Replay the journal file from start, rebuilding _byCmdId.  Called once at
  /// construction.  Tolerant of partial trailing lines (a crash mid-write
  /// leaves an unterminated line — we discard anything that fails to parse,
  /// since the design treats a non-COMPLETED state as "in flight" anyway).
  /// </summary>
  private void Replay()
  {
    if (!File.Exists(_path))
    {
      return;
    }

    try
    {
      foreach (var line in File.ReadAllLines(_path))
      {
        if (string.IsNullOrWhiteSpace(line))
        {
          continue;
        }
        try
        {
          var entry = JsonSerializer.Deserialize<JournalEntry>(line, JsonOpts);
          if (entry != null && !string.IsNullOrEmpty(entry.CmdId))
          {
            _byCmdId[entry.CmdId] = entry;  // last writer wins
          }
        }
        catch (JsonException)
        {
          // Truncated or corrupt trailing line — skip.  This is intentional:
          // a partial line means the writer crashed before fsync, so the
          // entry was never durably committed and SHOULD be ignored.
        }
      }
    }
    catch (IOException ex)
    {
      Console.Error.WriteLine(
        $"[Worker GPU:{_gpuId}] Journal replay IO error ({ex.Message}); starting fresh");
      _byCmdId.Clear();
    }
  }


  /// <summary>
  /// Append a STARTED entry and fsync.  Call BEFORE beginning execution of a
  /// new cmd_id so that on crash + restart the journal proves "this cmd_id was
  /// dispatched but not completed" → caller can decide to refuse a
  /// re-dispatch or hand the cmd_id to the RESUME reconciliation path.
  /// </summary>
  public void AppendStarted(string cmdId, object payload)
  {
    Append(cmdId, JournalEvent.STARTED, payload);
  }


  /// <summary>
  /// Append a COMPLETED entry and fsync.  CRITICAL: must be called before the
  /// RESULTS message is sent over the wire (see class docstring for the
  /// invariant).  After this returns, a worker crash + restart will see
  /// COMPLETED in the journal and a re-dispatched PLAY for the same cmd_id
  /// will return the cached result instead of replaying.
  /// </summary>
  public void AppendCompleted(string cmdId, object resultPayload)
  {
    Append(cmdId, JournalEvent.COMPLETED, resultPayload);
  }


  /// <summary>
  /// Append a FAILED entry and fsync.  Used when execution started but errored
  /// out (engine crash, OperationCanceledException, etc.).  The partial
  /// payload should carry whatever WDL/pentanomial we managed to accumulate
  /// before the failure so the orchestrator can treat the cmd_id as terminal
  /// (not re-dispatchable under the same id) without losing partial data.
  /// </summary>
  public void AppendFailed(string cmdId, string reason, object partialPayload)
  {
    var wrapped = new
    {
      reason,
      partial = partialPayload
    };
    Append(cmdId, JournalEvent.FAILED, wrapped);
  }


  /// <summary>
  /// Append a PROGRESS entry recording partial chunk progress.  Two write
  /// modes via <paramref name="durable"/>:
  ///   durable=false  — write + flush, no fsync.  Cheap (~10s of µs).  Used
  ///                    for the typical mid-chunk progress tick.  On hard
  ///                    crash this entry may be lost from the OS page cache,
  ///                    which is acceptable: PROGRESS is a hint, not durable
  ///                    state, and COMPLETED/FAILED are still durable.
  ///   durable=true   — write + flush + fsync.  Used for the LAST progress
  ///                    tick before COMPLETED so that on crash the journal
  ///                    has the most recent durable partial-progress fact
  ///                    available to RESUME.
  ///
  /// We do still update the in-memory dict either way — Replay() on restart
  /// is by file order, and a non-fsynced trailing line that survives is
  /// fine; one that doesn't is also fine (we treat its absence as "no
  /// PROGRESS recorded since STARTED" and the cmd_id remains in_progress).
  /// </summary>
  public void AppendProgress(string cmdId, int gamesPlayed, int ofTotal,
      int[] partialWdl, bool durable = false)
  {
    var payload = new
    {
      games_played = gamesPlayed,
      of_total = ofTotal,
      partial_wdl = partialWdl ?? Array.Empty<int>(),
    };
    Append(cmdId, JournalEvent.PROGRESS, payload, fsyncOnDisk: durable);
  }


  /// <summary>
  /// Look up the latest entry for a cmd_id.  Returns null if the cmd_id has
  /// never been seen by this worker (or was seen but the journal was rotated
  /// — rotation is not currently implemented).  Used by HandlePlayAsync for
  /// the idempotency check.
  /// </summary>
  public JournalEntry LookupCommand(string cmdId)
  {
    if (string.IsNullOrEmpty(cmdId))
    {
      return null;
    }
    lock (_lock)
    {
      return _byCmdId.TryGetValue(cmdId, out var entry) ? entry : null;
    }
  }


  /// <summary>
  /// Snapshot every cmd_id whose latest journal entry is within the recency
  /// window (default 30 min before the most recent ts).  Returned as a fresh
  /// list so the caller can iterate without holding the lock.
  ///
  /// "Recency" is measured against the newest ts on disk, not wall clock —
  /// this is robust to long-idle workers (a worker that's been idle 6 hours
  /// still reports its last batch of pre-idle entries) and to clock skew
  /// between worker and server.  RESUME_REPLY consumers only care that the
  /// snapshot is "everything since the gap began", not that it equals "the
  /// last 30 min of wall clock time".
  /// </summary>
  public List<JournalEntry> SnapshotRecent(double recencyWindowSeconds = 1800.0)
  {
    lock (_lock)
    {
      if (_byCmdId.Count == 0)
      {
        return new List<JournalEntry>();
      }
      double newestTs = double.MinValue;
      foreach (var entry in _byCmdId.Values)
      {
        if (entry.Ts > newestTs) newestTs = entry.Ts;
      }
      double cutoff = newestTs - recencyWindowSeconds;
      var snapshot = new List<JournalEntry>(_byCmdId.Count);
      foreach (var entry in _byCmdId.Values)
      {
        if (entry.Ts >= cutoff)
        {
          snapshot.Add(entry);
        }
      }
      return snapshot;
    }
  }


  /// <summary>
  /// Internal: serialize one JournalEntry to a line, append to file, flush,
  /// optionally fsync, and update the in-memory dict.  All under _lock so two
  /// concurrent callers are strictly serialized — important because the file
  /// is opened, written, and closed inside this method (we don't keep a
  /// long-lived FileStream because that would complicate fsync semantics
  /// across multiple connection handlers).
  ///
  /// <paramref name="fsyncOnDisk"/> defaults true — STARTED/COMPLETED/FAILED
  /// always need durable on-disk commit before returning so the
  /// fsync-before-send invariant holds.  PROGRESS entries pass false to skip
  /// the fsync for the typical mid-chunk tick (the runner passes true only
  /// on the FINAL pre-COMPLETED progress tick).
  /// </summary>
  private void Append(string cmdId, JournalEvent ev, object payload,
      bool fsyncOnDisk = true)
  {
    if (string.IsNullOrEmpty(cmdId))
    {
      // Calling Append with no cmd_id would corrupt the in-memory dict
      // (key="").  Callers should never do this — log loudly so we catch
      // protocol bugs in dev rather than silently dropping entries.
      Console.Error.WriteLine(
        $"[Worker GPU:{_gpuId}] Journal.Append called with empty cmd_id (event={ev}); skipping");
      return;
    }

    // Build the entry object.  We serialize the whole thing then re-parse
    // Payload as a JsonElement for the in-memory dict — this is so the cached
    // entry has the exact bytes we wrote, not a reference to the caller's
    // mutable object.
    var entryObj = new
    {
      ts = (DateTime.UtcNow - DateTime.UnixEpoch).TotalSeconds,
      cmd_id = cmdId,
      @event = ev.ToString(),
      payload
    };
    string line = JsonSerializer.Serialize(entryObj, JsonOpts);

    lock (_lock)
    {
      // Open in append mode, write, flush, fsync (when requested), close.
      // FileStream.Flush(true) forces both the userspace buffer AND the OS
      // page cache to durable storage — required for STARTED/COMPLETED/FAILED.
      // For non-durable PROGRESS we still call Flush(false) so the bytes
      // leave userspace and become visible on disk to a concurrent reader.
      using (var fs = new FileStream(
          _path,
          FileMode.Append,
          FileAccess.Write,
          FileShare.Read,
          bufferSize: 4096,
          options: fsyncOnDisk ? FileOptions.WriteThrough : FileOptions.None))
      {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(line + "\n");
        fs.Write(bytes, 0, bytes.Length);
        fs.Flush(fsyncOnDisk);  // Flush(true) = fsync; Flush(false) = userspace flush only
      }

      // Re-parse the line so the cached payload is a JsonElement (matches
      // what Replay() would produce on a fresh start).
      var parsed = JsonSerializer.Deserialize<JournalEntry>(line, JsonOpts);
      if (parsed != null)
      {
        _byCmdId[cmdId] = parsed;
      }
    }
  }
}
