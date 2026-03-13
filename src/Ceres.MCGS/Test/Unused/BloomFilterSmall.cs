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

#endregion

namespace Ceres.MCGS.MCTSNodes.Unused
{
  /// <summary>
  /// A small Bloom filter used to perform approximate set membership testing.
  /// </summary>
  public struct BloomFilterSmall
  {
    /// <summary>
    /// Total number of bits in the filter.
    /// Note that a test showed decreasing from 1024 to 512 increased 
    /// the false positive rate from about 7% to 17%.
    /// </summary>
    private const int FilterBits = 1024;

    /// <summary>
    /// Number of different hash functions to be used.
    /// Note that a test showed increasing from 3 to 4
    /// yielded almost no improvement in the false positive rate.
    /// </summary>
    private const int HashFunctions = 3;

    /// <summary>
    /// Size of the array (of ulong) needed to contain all the filter bits.
    /// </summary>
    private const int ArraySize = FilterBits / 64;

    /// <summary>
    /// Actual filter bits.
    /// </summary>
    private ulong[] bitArray;


    /// <summary>
    /// Initializes the filter to the starting state.
    /// </summary>
    public void Initialize()
    {
      bitArray = new ulong[ArraySize];
    }

    /// <summary>
    /// Initializes filter (starting from same state as another specified filter).
    /// </summary>
    /// <param name="startingFilter"></param>
    public void Initialize(in BloomFilterSmall startingFilter)
    {
      bitArray = GC.AllocateUninitializedArray<ulong>(ArraySize, pinned: false);
      for (int i = 0; i < ArraySize; i++)
      {
        // Make a clone of the array
        Array.Copy(startingFilter.bitArray, 0, bitArray, 0, ArraySize); 
      }
    }


    private static ulong MixHash(ulong x)
    {
      // A simple 64-bit hash mixer.
      unchecked
      {
        x = (x ^ x >> 30) * 0xbf58476d1ce4e5b9UL;
        x = (x ^ x >> 27) * 0x94d049bb133111ebUL;
        x ^= x >> 31;
        return x;
      }
    }


    /// <summary>
    /// Adds the given value to the Bloom filter.
    /// </summary>
    public void Add(ulong value)
    {
      // Create a base hash and increment
      ulong hash = MixHash(value);
      ulong delta = hash >> 17 | hash << 47; // rotate right 17

      // Apply k=HashFunctions times using double hashing.
      for (int i = 0; i < HashFunctions; i++)
      {
        SetBit((uint)(hash % FilterBits));
        hash += delta;
      }
    }


    /// <summary>
    /// Checks if the given value might be in the set;
    /// false indicates definitely not, whereas true is possibly yes.
    /// </summary>
    public bool ContainsPossibly(ulong value)
    {
      ulong hash = MixHash(value);
      ulong delta = hash >> 17 | hash << 47;

      for (int i = 0; i < HashFunctions; i++)
      {
        if (!TestBit((uint)(hash % FilterBits)))
        {
          return false;
        }
        hash += delta;
      }

      return true;
    }


    /// <summary>
    /// Sets the bit at the given index within the filter.
    /// </summary>
    private void SetBit(uint index)
    {
      int wordIndex = (int)(index >> 6);     // divide by 64
      int bitInWord = (int)(index & 0x3F);   // remainder mod 64
      ulong mask = 1UL << bitInWord;

      bitArray[wordIndex] |= mask;
    }


    /// <summary>
    /// Tests if the bit at the given index is set.
    /// </summary>
    private bool TestBit(uint index)
    {
      int wordIndex = (int)(index >> 6);
      int bitInWord = (int)(index & 0x3F);
      ulong mask = 1UL << bitInWord;

      return (bitArray[wordIndex] & mask) != 0;
    }


    public override bool Equals(object obj)
    {
      if (obj is BloomFilterSmall other)
      {
        return Equals(other);
      }
      return false;
    }


    public bool Equals(BloomFilterSmall other)
    {
      if (ReferenceEquals(bitArray, other.bitArray))
      {
        return true;
      }

      if (bitArray is null || other.bitArray is null || bitArray.Length != other.bitArray.Length)
      {
        return false;
      }

      for (int i = 0; i < bitArray.Length; i++)
      {
        if (bitArray[i] != other.bitArray[i])
        {
          return false;
        }
      }
      return true;
    }


    public override int GetHashCode()
    {
      if (bitArray is null)
      {
        return 0;
      }

      HashCode hash = new ();
      for (int i = 0; i < bitArray.Length; i++)
      {
        hash.Add(bitArray[i]);
      }

      return hash.ToHashCode();
    }

    public static bool operator ==(BloomFilterSmall left, BloomFilterSmall right) => left.Equals(right);

    public static bool operator !=(BloomFilterSmall left, BloomFilterSmall right) => !(left == right);    
  }

}
