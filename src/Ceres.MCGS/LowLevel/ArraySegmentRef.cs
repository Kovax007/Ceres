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
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Represents a reference to a segment of an array managed by a pool,
/// providing access to a contiguous region of elements of type T.
/// </summary>
public record struct ArraySegmentRef<T> where T : struct
{
  /// <summary>
  /// Represents a segment of an array from a pool managed by the parent."/>.
  /// </summary>
  private readonly ArraySegmentPool<T> owningPool;

  /// <summary>
  /// Index of the first item in the segment within the parent array.
  /// </summary>
  internal int startIndex;

  /// <summary>
  /// Number of items allocated in the segment.
  /// </summary>
  internal int numItemsAllocated;
  

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="owningPool"></param>
  /// <param name="start"></param>
  /// <param name="capacity"></param>
  internal ArraySegmentRef(ArraySegmentPool<T> owningPool, int start, int capacity)
  {
    this.owningPool = owningPool;
    startIndex = start;
    numItemsAllocated = capacity;
  }


  /// <summary>
  /// Returns the number of items used in the segment.
  /// </summary>
  public int NumItemsAllocated => numItemsAllocated;


  /// <summary>
  /// Returns a span over the logical portion of the segment.
  /// </summary>
  public Span<T> Span => MemoryMarshal.CreateSpan(ref owningPool.ItemAt(startIndex), numItemsAllocated);


  /// <summary>
  /// By-ref indexer over the logical portion of the segment.
  /// </summary>
  public ref T this[int index]
  {
    get
    {
#if DEBUG
      // Fast unsigned check handles both negative and too-large.
      if ((uint)index >= (uint)numItemsAllocated)
      {
        throw new ArgumentOutOfRangeException(nameof(index));
      }
#endif
      return ref owningPool.ItemAt(startIndex + index);
    }
  }


  /// <summary>
  /// Ensures that the segment has sufficient capacity for a specified number of items.
  /// </summary>
  /// <param name="neededItemCount"></param>
  public void EnsureSize(int neededItemCount)
  {
    if (neededItemCount <= numItemsAllocated)
    {
      return;
    }

    int newCap = ArraySegmentPool<T>.RoundUp(neededItemCount);
    ArraySegmentRef<T> bigger = owningPool.AllocateSegment(newCap);

    Span<T> oldData = Span.Slice(0, numItemsAllocated);
    oldData.CopyTo(bigger.Span);

    startIndex = bigger.startIndex;
    numItemsAllocated = bigger.numItemsAllocated;
  }
}
