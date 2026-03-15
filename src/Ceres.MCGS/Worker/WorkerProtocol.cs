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
  Init     = 0x01,
  Refit    = 0x02,
  Play     = 0x03,
  Stop     = 0x04,
  Status   = 0x05,
  Shutdown = 0x06,
  ProbeDeps = 0x07,
  Serialize = 0x08,
  NetVsNet  = 0x09
}


/// <summary>
/// Server-local configuration loaded from a worker_config.json at startup.
/// </summary>
public class WorkerLocalConfig
{
  public int    GpuId    { get; set; } = 0;
  public int    Port     { get; set; } = 5100;
  public string BindHost { get; set; } = "0.0.0.0";
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
/// </summary>
public class InitConfig
{
  public string EnginePath { get; set; }
  public string BookPath { get; set; }
  public int[] BatchSizes { get; set; }
  public bool UseCudaGraphs { get; set; } = true;
  public string OpponentExe { get; set; }
  public int OpponentNodes { get; set; }
  public int OpponentThreads { get; set; } = 2;
  public int EngineNodes { get; set; }
  public string NetPrefix { get; set; } = "";
  public string NetOptions { get; set; } = "";
  public string CeresJsonPath { get; set; }
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
/// Worker status response.
/// </summary>
public class WorkerStatus
{
  public string State { get; set; }  // "idle", "playing", "refitting"
  public string PerturbationId { get; set; }
  public int GamesPlayed { get; set; }
  public int[] WDL { get; set; }
  public int GpuId { get; set; }
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
      int dataBytes = numElements * 2;
      Half[] data = new Half[numElements];
      Buffer.BlockCopy(payload, offset, data, 0, dataBytes);
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

      // FP16 data
      byte[] data = new byte[kvp.Value.Length * 2];
      Buffer.BlockCopy(kvp.Value, 0, data, 0, data.Length);
      writer.Write(data);
    }

    return ms.ToArray();
  }
}
