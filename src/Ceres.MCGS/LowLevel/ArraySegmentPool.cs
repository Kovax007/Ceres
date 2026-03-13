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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Manages pool of items that can be allocated in blocks.
/// </summary>
/// <typeparam name="T"></typeparam>
public sealed class ArraySegmentPool<T> where T : struct
{
  /// <summary>
  /// Maximum total number of items.
  /// The underlying use of InlineArray necessitates a fixed maximum number of items.
  /// This number should be chosen to prevent overflow most scenarios.
  /// It will be larger for bigger batches and longer paths (deeper search).
  /// (also considering internal fragmentation due to allocation in blocks of GROWTH_QUANTUM).
  /// 
  /// However overflowing is not fatal because the search engine monitors for 
  /// batches approracing overflow and will stop the batch early if necessary to avoid.
  /// </summary>
  private const int MAX_ITEMS = 22 * 1024; 
                                           

  /// <summary>
  /// Number of items by which slot segments are grown at a time.
  /// N.B. Runtime speed is quite sensitive to this number.
  ///      For example, 6 is much better than 4 for large graphs (e.g. 10mm nodes).
  /// </summary>
  internal const int GROWTH_QUANTUM = 8;


  /// <summary>
  /// The inline array used to store items.
  /// </summary>
  [InlineArray(MAX_ITEMS)]
  private struct InlineBuffer { private T item0; }


  /// <summary>
  /// Inline fixed-size array of items.
  /// </summary>
  private InlineBuffer _buffer;


  /// <summary>
  /// Index of the next free item in the pool.
  /// </summary>
  private int nextFree;


  /// <summary>
  /// Allocates a new segment of the given number of items 
  /// (rounded up to the nearest multiple of GROWTH_QUANTUM).
  /// </summary>
  /// <param name="itemCount"></param>
  /// <returns></returns>
  public ArraySegmentRef<T> AllocateSegment(int? itemCount)
  {
    itemCount ??= GROWTH_QUANTUM;

    int capacity = RoundUp(itemCount.Value);

    int start = Interlocked.Add(ref nextFree, capacity) - capacity;
    
    if (start + capacity > MAX_ITEMS)
    {
      throw new InvalidOperationException("ArraySegmentPool overflow");
    }
    
    return new ArraySegmentRef<T>(this, start, capacity);
  }


  /// <summary>
  /// Returns a span representing a slice of the buffer.
  /// </summary>
  /// <param name="start"></param>
  /// <param name="length"></param>
  /// <returns></returns>
  internal Span<T> Slice(int start, int length)
  {
    Debug.Assert((uint)start < MAX_ITEMS && (uint)(start + length) <= MAX_ITEMS);

    ref T first = ref _buffer[start];
    //ref T first = ref Unsafe.Add(ref Unsafe.As<InlineBuffer, T>(ref _buffer), start);
    return MemoryMarshal.CreateSpan(ref first, length);
  }


  /// <summary>
  /// Returns a reference to the item at the given absolute index.
  /// </summary>
  /// <param name="absoluteIndex"></param>
  /// <returns></returns>
  internal ref T ItemAt(int absoluteIndex)
  {
    //Debug.Assert((uint)absoluteIndex < MAX_ITEMS);
    //return ref Unsafe.Add(ref Unsafe.As<InlineBuffer, T>(ref _buffer), absoluteIndex);
    return ref _buffer[absoluteIndex];
  }


  /// <summary>
  /// Returns the number of items currently allocated.
  /// </summary>
  public int Allocated => nextFree;


  /// <summary>
  /// Returns the fraction of the pool that is currently in use.
  /// </summary>
  public float FractionInUse => (float)Allocated / MAX_ITEMS;


  /// <summary>
  /// Clears the pool, releasing all allocated items.
  /// </summary>
  public void Clear(bool clearMem = true)
  {
    if (nextFree == 0)
    {
      return;
    }

    if (clearMem)
    {
      Slice(0, nextFree).Clear();
    }

    nextFree = 0;
  }


  /// <summary>
  /// Rounds up the given number to the nearest multiple of GROWTH_QUANTUM.
  /// </summary>
  /// <param name="n"></param>
  /// <returns></returns>
  internal static int RoundUp(int n)
  {
    const bool POWER_OF_TWO = (GROWTH_QUANTUM & (GROWTH_QUANTUM - 1)) == 0;

    if (POWER_OF_TWO)
    {
      return (n + GROWTH_QUANTUM - 1) & ~(GROWTH_QUANTUM - 1);
    }
    else
    {
      int q = GROWTH_QUANTUM;
      return ((n + q - 1) / q) * q;
    }
  }
}
