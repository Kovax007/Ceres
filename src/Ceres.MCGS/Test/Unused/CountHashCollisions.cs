
#region Using directives

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCGS.Storage.Structs;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Test
{
  public static class CountHashCollisions
  {
    public const string TAR1 = @"d:\tar\training-run1-test80-20230614-2317.tar";
    public const string TAR2 = @"d:\tar\training-run1-test80-20240531-1317.tar";
    public const string TAR3 = @"d:\tar\sepoct2024\training-run1-test80-20241029-2317.tar";

    
    const bool TEST_ONLY_64_BITS = false;


    public static void Test()
    {
      using (new TimingBlock("Hash collision search"))
      {
        Parallel.Invoke(
          () => CountHashCollisions.Count(CountHashCollisions.TAR1),
          () => CountHashCollisions.Count(CountHashCollisions.TAR2),
          () => CountHashCollisions.Count(CountHashCollisions.TAR3)
        );
      }
    }


    public static void Count(string tarFileName)
    {
      int collisionCount = 0;
      Dictionary<PosHash64, MGPosition> positionHashes = new();

      IEnumerable<PositionWithHistory> positions = new EncodedTrainingPositionReaderTAR(tarFileName)
          .EnumeratePositions()
          .Select(p => p.ToPositionWithHistory());

      Console.WriteLine("Begin collision search for " + tarFileName);
      foreach (PositionWithHistory testPos in positions)
      {
        PosHash64 hash = MGPositionHashing.Hash64(testPos.FinalPosMG);

        if (positionHashes.TryGetValue(hash, out MGPosition otherPos))
        {
          if (!otherPos.ToPosition.PiecesEqual(testPos.FinalPosMG.ToPosition))
          {
            Console.WriteLine("Collision detected for position: " + testPos.FinalPosMG);
            Console.WriteLine("Other position                 : " + otherPos);
            Console.WriteLine("Count " + positionHashes.Count);
            collisionCount++;
          }
        }
        positionHashes[hash] = testPos.FinalPosMG;
        if (positionHashes.Count % 1_000_000 == 0)
        {
          Console.WriteLine("Hash testing processed: " + positionHashes.Count);
        }
        continue;
      }
    }
  }
}
