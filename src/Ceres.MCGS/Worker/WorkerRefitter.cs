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

using Ceres.Chess.NNEvaluators.TensorRT;

#endregion

namespace Ceres.MCGS.Worker;

/// <summary>
/// Manages live weight refitting on TensorRT engines for the worker process.
///
/// For multi-profile engines, multiple TensorRTEngine handles share a single
/// underlying ICudaEngine. BatchRefitWeights() operates on the engine via any
/// handle, but InvalidateCudaGraphs() must be called on ALL handles.
/// </summary>
public class WorkerRefitter
{
  /// <summary>
  /// The engine handles (one per batch-size profile).
  /// BatchRefitWeights on [0] refits the shared engine; InvalidateCudaGraphs must hit all.
  /// </summary>
  private TensorRTEngine[] _engines;

  /// <summary>
  /// Reusable Half[] buffers keyed by weight name.
  /// Prevents .NET LOH fragmentation from allocating new arrays every refit.
  /// Buffers are allocated once on first use and reused for subsequent refits.
  /// </summary>
  private Dictionary<string, Half[]> _reusableBuffers;


  /// <summary>
  /// Initialize with the loaded multi-profile engine handles.
  /// </summary>
  public WorkerRefitter(TensorRTEngine[] engines)
  {
    _engines = engines ?? throw new ArgumentNullException(nameof(engines));
    if (engines.Length == 0) throw new ArgumentException("At least one engine handle required");
    _reusableBuffers = new Dictionary<string, Half[]>();
  }


  /// <summary>
  /// Discover fused dependency weight names for the given user weight names.
  /// Calls TRT_GetFusedDeps via the first engine handle — engine state is unchanged.
  /// Returns the JSON array string of fused dep names.
  /// </summary>
  public List<string> GetFusedDeps(IEnumerable<string> userWeightNames)
  {
    return _engines[0].GetFusedDeps(userWeightNames);
  }


  /// <summary>
  /// Refit the engine with new weights and invalidate CUDA graphs on all handles.
  ///
  /// The weight dictionary must contain ALL weights including fused dependencies
  /// (resolved by the Python orchestrator from the ONNX model).
  ///
  /// Returns a RefitResult with timing and success info.
  /// </summary>
  /// <summary>
  /// Get or allocate a reusable Half[] buffer for the given weight name and size.
  /// If the buffer already exists with the correct size, returns the existing one.
  /// This prevents LOH fragmentation from allocating new large arrays every refit.
  /// </summary>
  private Half[] GetOrAllocateBuffer(string name, int numElements)
  {
    if (_reusableBuffers.TryGetValue(name, out Half[] existing) && existing.Length == numElements)
    {
      return existing;
    }

    var buffer = new Half[numElements];
    _reusableBuffers[name] = buffer;
    return buffer;
  }


  public RefitResult Refit(string perturbationId, List<RefitWeightEntry> weightEntries)
  {
    var sw = Stopwatch.StartNew();

    try
    {
      // Copy weight data into reusable buffers to avoid LOH fragmentation.
      // Each refit receives ~29MB of Half[] arrays (32 weights for layer 13).
      // Without reuse, .NET allocates new LOH segments every iteration that are
      // never compacted, causing unbounded host memory growth.
      var weightsDict = new Dictionary<string, Half[]>(weightEntries.Count);
      foreach (var entry in weightEntries)
      {
        Half[] buffer = GetOrAllocateBuffer(entry.Name, entry.Data.Length);
        entry.Data.AsSpan().CopyTo(buffer);
        weightsDict[entry.Name] = buffer;
      }

      // Refit via any handle — they all share the same ICudaEngine
      int refittedCount = _engines[0].BatchRefitWeights(weightsDict);

      // Invalidate CUDA graphs on ALL handles (each has independent captured graphs)
      for (int i = 0; i < _engines.Length; i++)
      {
        _engines[i].InvalidateCudaGraphs();
      }

      sw.Stop();
      Console.WriteLine($"[WorkerRefitter] Refitted {refittedCount} weights for '{perturbationId}' in {sw.ElapsedMilliseconds}ms");

      return new RefitResult
      {
        Status = "refitted",
        PerturbationId = perturbationId,
        WeightsSet = refittedCount,
        ElapsedMs = sw.Elapsed.TotalMilliseconds
      };
    }
    catch (Exception ex)
    {
      sw.Stop();
      Console.Error.WriteLine($"[WorkerRefitter] Refit failed for '{perturbationId}': {ex.Message}");

      return new RefitResult
      {
        Status = "error",
        PerturbationId = perturbationId,
        Error = ex.Message,
        ElapsedMs = sw.Elapsed.TotalMilliseconds
      };
    }
  }
}
