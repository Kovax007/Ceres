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
using System.Threading;

#endregion

namespace Ceres.MCGS.LowLevel;

/// <summary>
/// Minimal, thread-safe dictionary built from N shards.
/// Each shard is a plain Dictionary guarded by a small lock.
/// Constructors interpret 'concurrencyLevel' as the requested shard count.
/// </summary>
public sealed class ShardedDictionary<TKey, TValue>
{
  private readonly Shard[] shards;
  private readonly IEqualityComparer<TKey> comparer;
  private readonly bool powerOfTwo;
  private readonly int mask; // valid only when _powerOfTwo==true

  // We only ever increment this in the API provided (no Remove present).
  private int _count;

  private sealed class Shard
  {
    public readonly Lock Gate = new ();
    public readonly Dictionary<TKey, TValue> Map;

    public Shard(int capacityPerShard, IEqualityComparer<TKey> comparer)
    {
      Map = new Dictionary<TKey, TValue>(capacityPerShard, comparer);
    }
  }


  /// <summary>
  /// new(concurrencyLevel: shardCount, capacity)
  /// </summary>
  public ShardedDictionary(int concurrencyLevel, int capacity)
    : this(concurrencyLevel, capacity, comparer: null)
  {
  }


  /// <summary>
  /// new(concurrencyLevel: shardCount, capacity, comparer)
  /// </summary>
  public ShardedDictionary(int concurrencyLevel, int capacity, IEqualityComparer<TKey> comparer)
  {
    if (concurrencyLevel <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(concurrencyLevel), "Shard count must be >= 1.");
    }
    ArgumentOutOfRangeException.ThrowIfNegative(capacity);

    this.comparer = comparer ?? EqualityComparer<TKey>.Default;

    int shardCount = concurrencyLevel;
    powerOfTwo = IsPowerOfTwo(shardCount);
    mask = powerOfTwo ? (shardCount - 1) : 0;

    shards = new Shard[shardCount];

    // Divide the requested capacity across shards (round up to avoid zeros).
    int perShard = shardCount == 0 ? 0 : Math.Max(1, (int)Math.Ceiling((double)capacity / shardCount));

    for (int i = 0; i < shardCount; i++)
    {
      shards[i] = new Shard(perShard, this.comparer);
    }

    _count = 0;
  }

  public int Count => Volatile.Read(ref _count);

  public TValue this[TKey key]
  {
    get
    {
      Shard s = ShardFor(key);
      lock (s.Gate)
      {
        if (s.Map.TryGetValue(key, out TValue value))
        {
          return value!;
        }
        throw new KeyNotFoundException();
      }
    }
    set
    {
      Shard s = ShardFor(key);
      lock (s.Gate)
      {
        if (!s.Map.TryAdd(key, value))
        {
          s.Map[key] = value;
        }
        else
        {
          Interlocked.Increment(ref _count);
        }
      }
    }
  }

  public bool TryGetValue(TKey key, out TValue value)
  {
    Shard s = ShardFor(key);
    lock (s.Gate)
    {
      return s.Map.TryGetValue(key, out value!);
    }
  }


  public bool TryAdd(TKey key, TValue value)
  {
    Shard s = ShardFor(key);
    lock (s.Gate)
    {
      if (s.Map.ContainsKey(key))
      {
        return false;
      }
      s.Map.Add(key, value);
      Interlocked.Increment(ref _count);
      return true;
    }
  }

  
  private Shard ShardFor(TKey key)
  {
    int h = comparer.GetHashCode(key);
    // Make non-negative without branching and distribute to shard index.
    if (powerOfTwo)
    {
      // (uint) cast avoids sign-propagating shifts; mask assumes power of two.
      int idx = (int)((uint)h) & mask;
      return shards[idx];
    }
    else
    {
      // Use unsigned modulo to avoid negative remainder for negative h.
      int idx = (int)((uint)h % (uint)shards.Length);
      return shards[idx];
    }
  }

  private static bool IsPowerOfTwo(int x)
  {
    return (x & (x - 1)) == 0;
  }
}
