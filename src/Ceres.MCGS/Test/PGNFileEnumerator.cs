#region Using directives

using System;
using System.Collections.Generic;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;


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
  /// <summary>
  /// Enumerators games/positions appearing in a PGN file.
  /// TODO: move this into Ceres project.
  /// </summary>
  public class PGNFileEnumerator
  {
    /// <summary>
    /// Name of underlying PGN file.
    /// </summary>
    public readonly string PGNFileName;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="pgnFileName"></param>
    /// <exception cref="Exception"></exception>
    public PGNFileEnumerator(string pgnFileName)
    {
      if (!System.IO.File.Exists(pgnFileName))
      {
        throw new Exception($"Specified PGN file not found: {pgnFileName}");
      }

      PGNFileName = pgnFileName;
    }


    /// <summary>
    /// Enumerates all positions in the PGN file as sequential Position objects.\
    /// optionally filtered by acceptFunc and/or a skip count.
    /// </summary>
    /// <param name="acceptFunc">if specified Position should be returned</param>
    /// <param name="skipCount">optional skip modulus to return only every N positions</param>
    /// <returns></returns>
    public IEnumerable<Position> EnumeratePositions(Predicate<Position> acceptFunc = null, int skipCount = -1)
    {
      Ceres.Chess.Textual.PgnFileTools.PgnStreamReader pgnReader = new();
      foreach (Ceres.Chess.Textual.PgnFileTools.GameInfo game in pgnReader.Read(PGNFileName))
      {
        game.Headers.TryGetValue("FEN", out string startFEN);
        MGPosition startPos = startFEN == null ? MGPosition.FromPosition(Position.StartPosition) : MGPosition.FromFEN(startFEN);
        MGPosition curPos = startPos;
        int plyIndex = 0;

        Position curPosAsPosition = curPos.ToPosition;

        if (acceptFunc == null || acceptFunc(curPosAsPosition))
        {
          yield return curPosAsPosition;
        }

        foreach (Ceres.Chess.Textual.PgnFileTools.Move move in game.Moves)
        {
          if (move.HasError)
          {
            Console.WriteLine("HasError " + move.Annotation);
            continue;
          }

          Move m1 = Move.FromSAN(curPos.ToPosition, move.ToAlgebraicString());
          MGMove mgMove = MGMoveConverter.MGMoveFromPosAndMove(curPos.ToPosition, m1);
          curPos.MakeMove(mgMove);
          curPosAsPosition = curPos.ToPosition;

          if (acceptFunc == null || acceptFunc(curPosAsPosition))
          {
            yield return curPosAsPosition;
          }

          if (skipCount == -1 || plyIndex % skipCount != 0)
          {
            plyIndex++;
          }
        }
      }
    }


    /// <summary>
    /// Enumerates sequence of all positions (with their full history within the game)
    /// optionally filtered by acceptFunc.
    /// </summary>
    /// <param name="acceptFunc"></param>
    /// <returns></returns>
    public IEnumerable<PositionWithHistory> EnumeratePositionWithHistory(Predicate<PositionWithHistory> acceptFunc = null)
    {
      foreach (var (game, positionIndex, curPositionAndMoves) in EnumeratePositionWithDetail(acceptFunc))
      {
        yield return curPositionAndMoves;
      }
    }


    /// <summary>
    /// Enumerates sequence of all positions (with their full history within the game)
    /// optionally filtered by acceptFunc. 
    /// Associated information (game info and position index) is also returned.
    /// </summary>
    /// <param name="acceptFunc"></param>
    /// <returns></returns>
    public IEnumerable<(Chess.Textual.PgnFileTools.GameInfo, int, PositionWithHistory)> EnumeratePositionWithDetail(Predicate<PositionWithHistory> acceptFunc = null)
    {
      Chess.Textual.PgnFileTools.PgnStreamReader pgnReader = new();
      foreach (Ceres.Chess.Textual.PgnFileTools.GameInfo game in pgnReader.Read(PGNFileName))
      {
        int positionIndex = 0;
        game.Headers.TryGetValue("FEN", out string startFEN);
        MGPosition startPos = startFEN == null ? MGPosition.FromPosition(Position.StartPosition) : MGPosition.FromFEN(startFEN);
        MGPosition curPos = startPos;
        PositionWithHistory curPositionAndMoves = new PositionWithHistory(startPos);

        if (acceptFunc == null || acceptFunc(curPositionAndMoves))
        {
          yield return (game, positionIndex++, curPositionAndMoves);
        }

        foreach (Ceres.Chess.Textual.PgnFileTools.Move move in game.Moves)
        {
          if (move.HasError)
          {
            Console.WriteLine("HasError " + move.Annotation);
            continue;
          }

          try
          {
            Move m1 = Move.FromSAN(curPos.ToPosition, move.ToAlgebraicString());
            MGMove mgMove = MGMoveConverter.MGMoveFromPosAndMove(curPos.ToPosition, m1);
            curPositionAndMoves.AppendMove(mgMove);
            curPos.MakeMove(mgMove);
          }
          catch (Exception)
          {
            Console.WriteLine($"Invalid move found in {PGNFileName} position {curPos.ToPosition.FEN} saw move string {move.ToString()}. Skipping position.");
          }

          if (acceptFunc == null || acceptFunc(curPositionAndMoves))
          {
            yield return (game, positionIndex++, curPositionAndMoves);
          }
        }
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