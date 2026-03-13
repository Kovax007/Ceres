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
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Ceres.MCGS.Graphs.GEdges;


#endregion

namespace Ceres.MCGS.Storage
{
  public static class AVXHelper
  {
    /// <summary>
    /// Searches for the first GEdgeStruct in the array (of length numEdges) whose F1 field equals f1MatchValue.
    /// Returns its index, or -1 if not found.
    /// </summary>
    public static unsafe int FindFirstMatchF1(GEdgeStruct* edges, int numEdges, int f1MatchValue)
    {
      throw new Exception("Make sure the ChildIndex is moved to offset 0 in GEdgeStruct, assumed here");

      // Each GEdgeStruct is 32 bytes, i.e. 8 ints.
      const int intsPerEdge = 8;

      if (Avx2.IsSupported)
      {
        // For F1 (offset 0), the int index for each record in a block of 8 is:
        // (0, 8, 16, 24, 32, 40, 48, 56)
        Vector256<int> offsetVector = Vector256.Create(0, 8, 16, 24, 32, 40, 48, 56);
        Vector256<int> targetVector = Vector256.Create(f1MatchValue);

        // Reinterpret the pointer as an int pointer.
        int* basePtr = (int*)edges;
        int i = 0;
        // Process blocks of 8 records at a time.
        for (; i <= numEdges - 8; i += 8)
        {
          // Compute the base index (in int elements) for the block.
          int baseIndex = i * intsPerEdge;
          Vector256<int> baseOffsetVector = Vector256.Create(baseIndex);
          // Compute indices for F1 field for each record in the block.
          Vector256<int> indices = Avx2.Add(offsetVector, baseOffsetVector);
          // Gather 8 F1 values; scale is 4 bytes (size of int).
          Vector256<int> f1Values = Avx2.GatherVector256(basePtr, indices, 4);

          // Compare the gathered F1 values with the target value.
          Vector256<int> cmp = Avx2.CompareEqual(f1Values, targetVector);

          // Convert the comparison result into a bitmask.
          int mask = Avx.MoveMask(cmp.AsSingle());
          if (mask != 0)
          {
            // Find the lane index of the first match.
            int lane = BitOperations.TrailingZeroCount(mask);
            return i + lane;
          }
        }

        // Process any remaining records with a scalar loop.
        for (int j = i; j < numEdges; j++)
        {
          // F1 is the first int of the record at index j.
          int value = *(basePtr + (j * intsPerEdge));
          if (value == f1MatchValue)
            return j;
        }
        return -1; // No match found.
      }
      else
      {
        // Fallback to scalar search if AVX2 is not supported.
        int* basePtr = (int*)edges;
        for (int j = 0; j < numEdges; j++)
        {
          int value = *(basePtr + (j * intsPerEdge));
          if (value == f1MatchValue)
            return j;
        }
        return -1;
      }
    }
  }
}
