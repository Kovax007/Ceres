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
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.MCGS.Worker;

/// <summary>
/// Command types for the worker protocol.
/// Wire format: [cmd:1byte][length:4bytes LE][payload:N bytes]
/// </summary>
public enum WorkerCommand : byte
{
  Init              = 0x01,
  Refit             = 0x02,
  Play              = 0x03,
  Stop              = 0x04,
  Status            = 0x05,
  Shutdown          = 0x06,
  ProbeDeps         = 0x07,  // Discover fused TRT weight dependencies (no engine state change)
  Serialize         = 0x08,  // Serialize current engine weights to a file on the worker host
  NetVsNet          = 0x09,  // Run a Ceres-vs-Ceres tournament between two networks
  ListPlayedOffsets = 0x0A,  // Snapshot of completed (opening_idx, r1, r2) game pairs in current PLAY
  Resume            = 0x0B,  // Server-on-reconnect handshake querying the worker journal
  Progress          = 0x0C   // Per-K-pair PROGRESS push from runner during PLAY
}


/// <summary>
/// Server-local configuration loaded from a worker_config.json at startup.
/// Contains everything needed to launch one worker instance: network identity
/// (gpu, port, bind address) plus server-specific paths (SF, book, Ceres.json).
///
/// Launch: Ceres.MCGS --worker --worker-config /path/to/worker_config_gpu0.json
/// Optional overrides: --gpu N  --port P  --host ADDR
/// </summary>
public class WorkerLocalConfig
{
  // Network identity
  public int    GpuId    { get; set; } = 0;
  public int    Port     { get; set; } = 5100;
  public string BindHost { get; set; } = "0.0.0.0";

  // Server-specific paths — used as fallback when INIT sends empty strings
  public string CeresJsonPath { get; set; } = "";
  public string OpponentExe   { get; set; } = "";
  public string BookPath      { get; set; } = "";

  public static WorkerLocalConfig Load(string path)
  {
    string json = File.ReadAllText(path);
    return JsonSerializer.Deserialize<WorkerLocalConfig>(json, new JsonSerializerOptions
    {
      PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
      ReadCommentHandling  = JsonCommentHandling.Skip,
      AllowTrailingCommas  = true,
    }) ?? new WorkerLocalConfig();
  }
}


/// <summary>
/// Configuration sent with the INIT command.
/// Server-local paths (BookPath, OpponentExe, CeresJsonPath) are optional here —
/// the worker falls back to its WorkerLocalConfig if these arrive empty.
/// </summary>
public class InitConfig
{
  public string EnginePath { get; set; }
  public string BookPath { get; set; } = "";
  public int[] BatchSizes { get; set; }
  public bool UseCudaGraphs { get; set; } = true;
  public string OpponentExe { get; set; } = "";
  public int OpponentNodes { get; set; }
  public int OpponentThreads { get; set; } = 2;
  public int EngineNodes { get; set; }
  public string NetPrefix { get; set; } = "";
  public string NetOptions { get; set; } = "";
  public string CeresJsonPath { get; set; } = "";
  public string TablesDir { get; set; }
  public Dictionary<string, double> SearchParams { get; set; }
}


/// <summary>
/// Configuration sent with the PLAY command.
/// </summary>
public class PlayConfig
{
  public string PerturbationId { get; set; }
  public int NumGamePairs { get; set; }
  public int OpeningOffset { get; set; }
  public int Concurrency { get; set; }

  /// <summary>
  /// Opening book shuffle seed.  -1 = Randomize (default, no seed).
  /// >= 0 = ShuffleDeterministic with this seed.
  /// Antithetical pairs sharing the same seed play the same openings (CRN).
  /// </summary>
  public int OpeningSeed { get; set; } = -1;

  /// <summary>
  /// Server-assigned command id for idempotent dispatch.  Format produced by
  /// the orchestrator: "{iter}-{rnd}-{variant}-chunk{idx}-{uuid8}".
  ///
  /// Backwards compat: nullable.  Older Python clients omit this field — the
  /// worker MUST continue to operate as before when CmdId is null/empty (no
  /// journal entry, just a debug log line).  The value is echoed back
  /// unchanged in the matching TournamentResult so the orchestrator can
  /// reconcile result→dispatch.
  /// </summary>
  public string CmdId { get; set; }
}


/// <summary>
/// Result of a completed game pair, streamed back during PLAY.
/// </summary>
public class GamePairResult
{
  public string Type { get; set; } = "game_pair";
  public string PerturbationId { get; set; }
  public int OpeningIdx { get; set; }
  public int R1 { get; set; }  // +1=win, 0=draw, -1=loss for Ceres
  public int R2 { get; set; }
  public int[] CumulativeWDL { get; set; }  // [wins, draws, losses]
}


/// <summary>
/// Final result of a tournament (complete or stopped early).
/// </summary>
public class TournamentResult
{
  public string Type { get; set; } = "tournament_done";
  public string PerturbationId { get; set; }
  public int Wins { get; set; }
  public int Draws { get; set; }
  public int Losses { get; set; }
  public int GamesPlayed { get; set; }
  public int[] Pentanomial { get; set; }  // [WW, WD, WL, DD, LD, LL]

  /// <summary>
  /// Number of game-thread tasks that exited via an exception during this
  /// tournament.  Zero on a clean run.  Non-zero means GamesPlayed is below
  /// the requested NumGamePairs * 2 because one or more concurrency threads
  /// died early — the wins/draws/losses are from the surviving threads only.
  /// </summary>
  public int FailedThreads { get; set; } = 0;

  /// <summary>
  /// First few exception messages from the failed threads (capped server-side
  /// at 8 entries).  Empty when FailedThreads == 0.  Surfaced so the orchestrator
  /// can log a precise reason for any partial result instead of guessing.
  /// </summary>
  public string[] FailureReasons { get; set; } = System.Array.Empty<string>();

  /// <summary>
  /// Echo of the server-assigned CmdId from the originating PLAY.
  /// Nullable for backwards compat: this build always echoes the value back;
  /// older clients ignore the field.  When the orchestrator receives a
  /// RESULTS message with CmdId == null (legacy worker behaviour), it MUST
  /// treat the result as belonging to the most recently dispatched CmdId
  /// for that worker — see WorkerOrchestrator.dispatched_cmd_ids handling.
  /// </summary>
  public string CmdId { get; set; }
}


/// <summary>
/// Request payload for the PROBE_DEPS command (JSON).
/// </summary>
public class ProbeDepsRequest
{
  public List<string> WeightNames { get; set; }
}


/// <summary>
/// Response for the PROBE_DEPS command.
/// </summary>
public class ProbeDepsResult
{
  public string Status { get; set; }         // "ok" or "error"
  public List<string> FusedDeps { get; set; }
  public int UserWeights { get; set; }        // echo back count for sanity
  public string Error { get; set; }
}


/// <summary>
/// Request payload for the SERIALIZE command (JSON).
/// </summary>
public class SerializeRequest
{
  public string OutputPath { get; set; }
}


/// <summary>
/// Response for the SERIALIZE command.
/// </summary>
public class SerializeResult
{
  public string Status { get; set; }   // "ok" or "error"
  public string OutputPath { get; set; }
  public long SizeBytes { get; set; }
  public string Error { get; set; }
}


/// <summary>
/// Configuration sent with the NETVSNET command.
/// Runs a Ceres-vs-Ceres tournament between two networks.
/// </summary>
public class NetVsNetConfig
{
  public string Net1Path { get; set; }        // ONNX/engine path on worker filesystem
  public string Net1Options { get; set; }     // e.g. "cudagraphs=true;BF16=true;V1TEMP=0.6989"
  public string Net1Prefix { get; set; }      // e.g. "ONNX_TRT:" for LC0 nets, "" for Ceres nets
  public string Net2Path { get; set; }
  public string Net2Options { get; set; }
  public string Net2Prefix { get; set; }
  public int NodesPerMove { get; set; }       // same for both engines
  public int NumGamePairs { get; set; }
  public int Concurrency { get; set; }
  public Dictionary<string, double> SearchParams { get; set; }  // applied to both engines
  public int OpeningOffset { get; set; }      // starting index into shuffled openings (chunked dispatch)
  public int OpeningSeed { get; set; } = -1;  // >=0 → ShuffleDeterministic; -1 → Randomize
}


/// <summary>
/// Weight entry for the REFIT command binary payload.
/// </summary>
public struct RefitWeightEntry
{
  public string Name;
  public Half[] Data;
}


/// <summary>
/// Result of a refit operation.
/// </summary>
public class RefitResult
{
  public string Status { get; set; }
  public string PerturbationId { get; set; }
  public int WeightsSet { get; set; }
  public double ElapsedMs { get; set; }
  public string Error { get; set; }
}


/// <summary>
/// Compact per-pair entry used by ListPlayedOffsets: no perturbation_id (echoed
/// once at the top level), just the triple we need to replay into the dispatcher.
/// </summary>
public class PlayedOffsetEntry
{
  public int OpeningIdx { get; set; }
  public int R1 { get; set; }
  public int R2 { get; set; }
}


/// <summary>
/// Response to LIST_PLAYED_OFFSETS: snapshot of every completed game pair in
/// the CURRENT PlayAsync call (cleared on each new PLAY via _gamePairsByOpening.Clear()).
/// Used by orchestrator reconnect flow to recover game_pair events emitted while
/// the stream was dropped, so a pair doesn't need to be replayed by rescue workers.
/// </summary>
public class PlayedOffsetsResult
{
  public string Type { get; set; } = "played_offsets";
  public string PerturbationId { get; set; }
  public string State { get; set; }  // worker state at time of snapshot
  public List<PlayedOffsetEntry> Offsets { get; set; } = new();
}


/// <summary>
/// Entry in a RESUME request: one cmd_id the server expects to know about,
/// plus the state the server *believes* it's in.  The worker uses
/// ExpectedState only as a hint for diagnostics — its own journal is
/// authoritative for whatever state the cmd_id is actually in.
/// </summary>
public class ResumeServerViewEntry
{
  public string CmdId { get; set; }
  public string ExpectedState { get; set; }  // "in_progress" | "completed" | etc.
}


/// <summary>
/// RESUME request payload (server → worker), sent immediately after a TCP
/// reconnect and BEFORE any subsequent PLAY.  Contains the orchestrator's
/// view of which cmd_ids the worker should know about.  The worker walks its
/// journal and answers with a ResumeReply that may include cmd_ids the
/// server did not list (e.g. results that completed during the disconnect
/// window).
/// </summary>
public class ResumeRequest
{
  public List<ResumeServerViewEntry> ServerView { get; set; } = new();
}


/// <summary>
/// Single entry in a RESUME_REPLY journal slice.  When State == "completed",
/// Results is populated with the cached TournamentResult payload the worker
/// durably recorded at COMPLETED time.  For "started" / "failed" Results is
/// null (only the state matters; failures may include a partial payload but
/// the wire shape is kept simple — orchestrator just needs the terminality
/// signal to requeue).
/// </summary>
public class ResumeReplyEntry
{
  public string CmdId { get; set; }
  public string State { get; set; }   // "started" | "completed" | "failed"
  public TournamentResult Results { get; set; }
}


/// <summary>
/// RESUME_REPLY (worker → server), sent in response to a RESUME request.
/// Includes:
///   - every cmd_id the server asked about (state from journal, results if
///     COMPLETED).
///   - every cmd_id the worker has touched within a recency window (~30 min)
///     that the server did NOT ask about — this is how the orchestrator
///     ingests results that completed while the server was disconnected.
/// </summary>
public class ResumeReply
{
  public string Type { get; set; } = "resume_reply";
  public string Status { get; set; } = "ok";
  public List<ResumeReplyEntry> Journal { get; set; } = new();
}


/// <summary>
/// Async PROGRESS push (worker → server) emitted by WorkerTournamentRunner
/// every K game-pairs during a PLAY.  Lets the orchestrator track partial
/// chunk progress so a hard-crashed worker bounds lost work to K pairs
/// instead of the whole chunk.
///
/// Wire shape: standard JSON-with-length-prefix response, type="progress".
/// PartialWdl is [wins, draws, losses] from Ceres perspective accumulated
/// since the start of the chunk.  Send is best-effort: the runner wraps it
/// in try/catch so a dead socket can't crash the play loop.
///
/// Backwards compat: older orchestrators that don't recognise type="progress"
/// fall through to the "unknown message type" log branch in send_play and
/// continue waiting for the next response — no behaviour change required.
/// </summary>
public class ProgressMessage
{
  public string Type { get; set; } = "progress";
  public string CmdId { get; set; }
  public int GamesPlayed { get; set; }
  public int OfTotal { get; set; }
  public int[] PartialWdl { get; set; }
}


/// <summary>
/// Worker status response.
/// </summary>
public class WorkerStatus
{
  public string State { get; set; }  // "idle", "playing", "refitting", "uninitialized"
  public string PerturbationId { get; set; }
  public int GamesPlayed { get; set; }
  public int[] WDL { get; set; }
  public int GpuId { get; set; }
  public int[] Pentanomial { get; set; }  // [WW, WD, WL, DD, LD, LL], null until first pair completes
}


/// <summary>
/// Handles reading and writing messages on the worker TCP protocol.
///
/// Wire format per message:
///   [command: 1 byte] [payload_length: 4 bytes LE] [payload: N bytes]
///
/// For REFIT, the payload is binary:
///   [perturbation_id: UTF-8 + \0] [num_weights: int32 LE]
///   then for each weight:
///     [name: UTF-8 + \0] [num_elements: int32 LE] [data: num_elements * 2 bytes (FP16)]
///
/// For other commands, the payload is JSON.
/// Responses are always JSON prefixed with [length: 4 bytes LE].
/// </summary>
public static class WorkerProtocol
{
  static readonly JsonSerializerOptions JsonOpts = new()
  {
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    WriteIndented = false
  };


  /// <summary>
  /// Read a command header (cmd + length) from the stream.
  /// Returns false on EOF/disconnect.
  /// </summary>
  public static async Task<(WorkerCommand cmd, int length)?> ReadCommandHeaderAsync(
      Stream stream, CancellationToken ct = default)
  {
    byte[] header = new byte[5];
    int totalRead = 0;
    while (totalRead < 5)
    {
      int read = await stream.ReadAsync(header.AsMemory(totalRead, 5 - totalRead), ct);
      if (read == 0) return null;  // EOF
      totalRead += read;
    }

    WorkerCommand cmd = (WorkerCommand)header[0];
    int length = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(1));
    return (cmd, length);
  }


  /// <summary>
  /// Read exactly N bytes from the stream.
  /// </summary>
  public static async Task<byte[]> ReadExactAsync(Stream stream, int count, CancellationToken ct = default)
  {
    byte[] buffer = new byte[count];
    int totalRead = 0;
    while (totalRead < count)
    {
      int read = await stream.ReadAsync(buffer.AsMemory(totalRead, count - totalRead), ct);
      if (read == 0) throw new EndOfStreamException("Connection closed while reading payload");
      totalRead += read;
    }
    return buffer;
  }


  /// <summary>
  /// Read exactly N bytes from the stream into a pre-allocated buffer.
  /// Avoids LOH allocations for large payloads (e.g., ~29MB REFIT data).
  /// </summary>
  public static async Task ReadExactIntoAsync(Stream stream, byte[] buffer, int count, CancellationToken ct = default)
  {
    int totalRead = 0;
    while (totalRead < count)
    {
      int read = await stream.ReadAsync(buffer.AsMemory(totalRead, count - totalRead), ct);
      if (read == 0) throw new EndOfStreamException("Connection closed while reading payload");
      totalRead += read;
    }
  }


  /// <summary>
  /// Parse a JSON payload into the specified type.
  /// </summary>
  public static T ParseJson<T>(byte[] payload) =>
      JsonSerializer.Deserialize<T>(payload, JsonOpts);


  /// <summary>
  /// Parse the binary REFIT payload into perturbation ID + weight entries.
  /// </summary>
  public static (string perturbationId, List<RefitWeightEntry> weights) ParseRefitPayload(byte[] payload)
  {
    int offset = 0;

    // Read perturbation_id (null-terminated UTF-8)
    int nullPos = Array.IndexOf(payload, (byte)0, offset);
    if (nullPos < 0) throw new FormatException("Missing null terminator for perturbation_id");
    string perturbationId = Encoding.UTF8.GetString(payload, offset, nullPos - offset);
    offset = nullPos + 1;

    // Read num_weights (int32 LE)
    int numWeights = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset));
    offset += 4;

    var weights = new List<RefitWeightEntry>(numWeights);
    for (int i = 0; i < numWeights; i++)
    {
      // Read name (null-terminated UTF-8)
      nullPos = Array.IndexOf(payload, (byte)0, offset);
      if (nullPos < 0) throw new FormatException($"Missing null terminator for weight name at index {i}");
      string name = Encoding.UTF8.GetString(payload, offset, nullPos - offset);
      offset = nullPos + 1;

      // Read num_elements (int32 LE)
      int numElements = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset));
      offset += 4;

      // Read FP16 data (numElements * 2 bytes)
      // Half is not a primitive type so Buffer.BlockCopy won't work — use MemoryMarshal instead.
      int dataBytes = numElements * 2;
      Half[] data = new Half[numElements];
      payload.AsSpan(offset, dataBytes).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()));
      offset += dataBytes;

      weights.Add(new RefitWeightEntry { Name = name, Data = data });
    }

    return (perturbationId, weights);
  }


  /// <summary>
  /// Send a JSON response prefixed with its length.
  /// </summary>
  public static async Task SendResponseAsync<T>(Stream stream, T response, CancellationToken ct = default)
  {
    byte[] json = JsonSerializer.SerializeToUtf8Bytes(response, JsonOpts);
    byte[] lengthPrefix = new byte[4];
    BinaryPrimitives.WriteInt32LittleEndian(lengthPrefix, json.Length);

    await stream.WriteAsync(lengthPrefix, ct);
    await stream.WriteAsync(json, ct);
    await stream.FlushAsync(ct);
  }


  /// <summary>
  /// Build a binary REFIT payload from perturbation ID and weight dictionary.
  /// Used by the Python-side orchestrator (or C# test client).
  /// </summary>
  public static byte[] BuildRefitPayload(string perturbationId, Dictionary<string, Half[]> weights)
  {
    using var ms = new MemoryStream();
    using var writer = new BinaryWriter(ms);

    // perturbation_id + null terminator
    writer.Write(Encoding.UTF8.GetBytes(perturbationId));
    writer.Write((byte)0);

    // num_weights
    byte[] countBytes = new byte[4];
    BinaryPrimitives.WriteInt32LittleEndian(countBytes, weights.Count);
    writer.Write(countBytes);

    foreach (var kvp in weights)
    {
      // name + null terminator
      writer.Write(Encoding.UTF8.GetBytes(kvp.Key));
      writer.Write((byte)0);

      // num_elements
      byte[] elemBytes = new byte[4];
      BinaryPrimitives.WriteInt32LittleEndian(elemBytes, kvp.Value.Length);
      writer.Write(elemBytes);

      // FP16 data — Half is not a primitive, use MemoryMarshal
      byte[] data = new byte[kvp.Value.Length * 2];
      System.Runtime.InteropServices.MemoryMarshal.AsBytes(kvp.Value.AsSpan()).CopyTo(data);
      writer.Write(data);
    }

    return ms.ToArray();
  }
}
