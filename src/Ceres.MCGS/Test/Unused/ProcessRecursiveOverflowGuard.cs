#region Using directives

using System.IO;


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

namespace Ceres.Train
{
  using System;
  using System.IO.MemoryMappedFiles;
  using System.Threading;

  public static class ProcessRecursiveOverflowGuard
  {
    private const string SharedMemoryName = "Global\\CeresProcessCounter";
    private const string MutexName = "Global\\CeresProcessCounterMutex";
    private const int MaxCeresExecutables = 8;

    public static void CheckRecursiveOverflow()
    {
      bool createdNew;
      using Mutex mutex = new Mutex(false, MutexName, out createdNew);
      try
      {
        mutex.WaitOne();

        using MemoryMappedFile mmf = MemoryMappedFile.CreateOrOpen(SharedMemoryName, sizeof(int));
        using MemoryMappedViewAccessor accessor = mmf.CreateViewAccessor();

        int count;
        accessor.Read(0, out count);
        count++;

        if (count > MaxCeresExecutables)
        {
          Console.WriteLine("Shutting down, possible infinite process recursion: too many instances running.");
          Environment.Exit(3);
        }

        accessor.Write(0, count);
      }
      catch (AbandonedMutexException)
      {
        Console.WriteLine("Warning: Mutex was abandoned. Continuing anyway.");
        // mutex is still acquired, save to proceed
      }
      finally
      {
        mutex.ReleaseMutex();
      }
    }

    /// <summary>
    /// Should be called once when the process is exiting.
    /// </summary>
    public static void ModifyCounter(int decrementCount)
    {
      bool createdNew;
      using Mutex mutex = new Mutex(false, MutexName, out createdNew);
      try
      {
        mutex.WaitOne();

        using MemoryMappedFile mmf = MemoryMappedFile.OpenExisting(SharedMemoryName);
        using MemoryMappedViewAccessor accessor = mmf.CreateViewAccessor();

        int count;
        accessor.Read(0, out count);
        count = Math.Max(0, count - decrementCount);
        accessor.Write(0, count);
      }
      catch (FileNotFoundException)
      {
        // Safe to ignore — nothing to decrement
      }
      finally
      {
        mutex.ReleaseMutex();
      }
    }
  }
}


#if NOT
    public static void MakeMoveTest()
    {
      MGPosition mg = Position.StartPosition.ToMGPosition;

      Position pos = Position.StartPosition;
      Chess.Move m1 = Chess.Move.FromSAN(mg.ToPosition, "e4");

      while (false)
        Benchmarking.DumpOperationTimeAndMemoryStats(() => { Position p = pos; p.MakeMove(m1); }, "conv");
//      Benchmarking.DumpOperationTimeAndMemoryStats(() => MGChessPositionConverter.PositionFromMGChessPosition(in mg), "conv");

            PGNFileEnumerator pgn = new(@"z:\chess\data\pgn\ccrl.pgn");// xxx
//      PGNFileEnumerator pgn = new(@"z:\chess\data\pgn\TCEC_Season_25_-_Frd_1_Final_League.pgn");

      foreach ((Chess.Textual.PgnFileTools.GameInfo gameInfo, int posIndex, PositionWithHistory pp) in pgn.EnumeratePositionWithDetail())
      {
        Position startPos = pp.InitialPosition;
        MGPosition startPosMG = startPos.ToMGPosition;
        Position priorPos;
        foreach (MGMove moveMG in pp.Moves)
        {
          if (false)
          {
            const string BAD = "b4rkr/5p2/N1p1p1q1/1p6/P1pP3P/4R1P1/1P1Q1P2/R4K2 w - - 3 27";
            startPos = Position.FromFEN(BAD);
            Move moveX = new Move(Move.MoveType.MoveCastleShort);
            startPos.MakeMove(moveX);
            Console.WriteLine(startPos.FEN);
          }

          if (startPos.PieceCount != startPosMG.PieceCount)
            throw new NotImplementedException();
          priorPos = startPos;
          Move move = MGMoveConverter.ToMove(moveMG);

//          Console.WriteLine(startPos.FEN + " " + moveMG + " " + move);

//          if (startPos.FEN.StartsWith("rnbqk2r/pp2ppbp/6p1/2p5/3PP3/2P2N2/P4PPP/1RBQKB1R b Kkq"))
//            Console.WriteLine("here");

          startPos.MakeMove(move);
          startPosMG.MakeMove(moveMG);

//          if (!startPos.PiecesEqual(startPosMG.ToPosition))
            bool fensSame = startPos.FEN == startPosMG.ToPosition.FEN;
            if (!fensSame)
            {
              Console.WriteLine();
              Console.WriteLine(startPos.FEN);
              Console.WriteLine(startPos.FEN + " " + move);
              Console.WriteLine(startPosMG.ToPosition.FEN + " " + moveMG);
              Console.WriteLine(fensSame);
            }
            //throw new Exception("bad");
          else
          {
            Console.Write(".");
          }
        }
      }

    }
#endif
