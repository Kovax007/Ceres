#region Using directives

using System;
using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;

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

namespace Ceres.MCGS.Test;

/// <summary>
/// Benchmark utilities for testing multi-GPU backend performance.
/// </summary>
public static class BackendMultiGPUBenchmark
{
  private static EncodedPositionBatchFlat BuildBatch(int batchSize, bool flip = false)
  {

    EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;
    EncodedPositionBatchBuilder batchBuilder = new(batchSize, NNEvaluator.InputTypes.All);

    for (int i = 0; i < batchSize; i++)
    {
      Position testPos = i % 2 == (flip ? 1 : 0) ? Position.StartPosition
                                                 : Position.FromFEN("4k3/1P3p2/8/7p/rB2pb2/2P4P/2K5/6R1 b - - 0 50");
      batchBuilder.Add(testPos);
    }
    EncodedPositionBatchFlat batch = batchBuilder.GetBatch();
    return batch;
  }


  public static void SingleBatchTest(bool useEvaluateIntoBuffers)
  {
    //const string NET = "C1-256-10-i8|cudagraphs=false";
    const string NET = "~T3_DISTILL_512_15_FP16_TRT";
    const string DEV = "TensorRT16";
    const string DEVICE1 = "GPU:0#" + DEV;
    const string DEVICE2 = "GPU:0#" + DEV;

    NNEvaluator evaluatorMulti = NNEvaluator.FromSpecification(NET, DEVICE1);
    NNEvaluator evaluatorSingle = NNEvaluator.FromSpecification(NET, DEVICE2);

    Console.WriteLine();
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "SINGLE VS MULTI COMPARISON");
    Console.WriteLine("  " + NET + ":   " + DEVICE1 + " vs " + DEVICE2);
    Console.WriteLine("Evaluate into buffers? " + useEvaluateIntoBuffers);
    Console.WriteLine();

    Console.WriteLine("  Batch        MultiMem      SingleMem     MultiMs SingleMs   Rel    Validation");

    for (int ix = 0; ix < 3; ix++)
    {
      for (int bs = 16; bs <= 384; bs += 16)
      {
        EncodedPositionBatchFlat batch = BuildBatch(bs);

        int countSingle = 0;
        int countMulti = 0;

        double sumSingle = 0;
        double sumMulti = 0;

        long memSingle = 0;
        long memMulti = 0;

        int dummyInt = 0;
        double dummyDouble = 0;
        long dummyLong = 0;

        // warmup
        BenchEvaluator(evaluatorSingle, batch, useEvaluateIntoBuffers, ref dummyInt, ref dummyDouble, ref dummyLong);
        BenchEvaluator(evaluatorMulti, batch, useEvaluateIntoBuffers, ref dummyInt, ref dummyDouble, ref dummyLong);

        IPositionEvaluationBatch resultSingle;
        IPositionEvaluationBatch resultMulti;

        //      Thread.Sleep(100);
        resultSingle = BenchEvaluator(evaluatorSingle, batch, useEvaluateIntoBuffers, ref countSingle, ref sumSingle, ref memSingle);
        //      Thread.Sleep(100);
        resultMulti = BenchEvaluator(evaluatorMulti, batch, useEvaluateIntoBuffers, ref countMulti, ref sumMulti, ref memMulti);

        if (resultSingle != null && resultMulti != null)
        {
          for (int i = 0; i < batch.NumPos; i++)
          {
            if (System.Math.Abs(resultSingle.GetV(i) - resultMulti.GetV(i)) > 0.01)
            {
              Console.WriteLine("bad v " + i + " " + resultSingle.GetV(i) + " " + resultMulti.GetV(i));
            }

            if (System.Math.Abs(resultSingle.PolicyRef(i).Entropy - resultMulti.PolicyRef(i).Entropy) > 0.05)
            {
              Console.WriteLine("bad p " + i + " " + resultSingle.PolicyRef(i) + "  VERUS  " + resultMulti.PolicyRef(i));
            }
          }
        }

        long memDiff = memMulti - memSingle;
        double relative = (sumMulti / countMulti) / (sumSingle / countSingle);
        Console.WriteLine($"  {bs,6:N0}   {memMulti,12:N0}  {memSingle,12:N0}     {1000 * sumMulti / countMulti,7:F2}ms  {1000 * sumSingle / countSingle,7:F2}ms  {relative,6:F3}x     OK");
      }
    }
  }


  private static IPositionEvaluationBatch BenchEvaluator(NNEvaluator evaluator,
                                                         EncodedPositionBatchFlat batch,
                                                         bool useEvaluateIntoBuffers,
                                                         ref int countSingle, ref double sumSingle,
                                                         ref long memory)
  {
    IPositionEvaluationBatch result = null;
    TimingStats tb = new();
    using (new TimingBlock(tb, target: TimingBlock.LoggingType.None))
    {
      long startMem = GC.GetTotalAllocatedBytes();
      if (useEvaluateIntoBuffers)
      {
        result = evaluator.EvaluateIntoBuffers(batch);
      }
      else
      {
        evaluator.EvaluateBatch(batch);
      }

      long endMem = GC.GetTotalAllocatedBytes();
      memory += endMem - startMem;
    }

    sumSingle += tb.ElapsedTimeSecs;
    countSingle++;

    return result;
  }
}
