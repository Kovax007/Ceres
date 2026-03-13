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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Coordinates running one or more tasks and waiting for their completion,
/// while tracking aggregate statistics about wait times.
/// </summary>
public class TaskCounter : IDisposable
{
  /// <summary>
  /// Synchronization primitive used to signal when all tasks are done.
  /// </summary>
  private readonly ManualResetEventSlim doneEvent;

  #region Aggregate wait time statistics

  /// <summary>
  /// The total number of tasks that have been started but not yet completed.
  /// </summary>
  private int activeTaskCount;

  /// <summary>
  /// Tracks the number of times a wait operation has been called.
  /// </summary>
  private long waitCallCount = 0;

  /// <summary>
  /// Represents the total accumulated wait time in milliseconds.
  /// </summary>
  private double totalWaitTimeMS = 0;

  /// <summary>
  /// Maximum wait time observed in milliseconds.
  /// </summary>
  private double maxWaitTimeMS = 0;

  /// <summary>
  /// Second highest wait time observed in milliseconds.
  /// </summary>
  private double secondMaxWaitTimeMS = 0;

  /// <summary>
  /// Represents the cumulative sum of the squares of wait times in milliseconds.
  /// </summary>
  private double sumOfSquaresWaitTimeMS = 0;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  public TaskCounter() =>  doneEvent = new ManualResetEventSlim(true);


  /// <summary>
  /// Runs the specified action as a task with state parameter, incrementing the count of active tasks.
  /// This signature avoids closure allocation by using a state parameter.
  /// </summary>
  /// <param name="action">The action to execute, taking a state parameter</param>
  /// <param name="state">The state object to pass to the action</param>
  public void Run(Action<object> action, object state)
  {
    Interlocked.Increment(ref activeTaskCount);
    doneEvent.Reset();

    Task.Run(() =>
    {
      // Need to explicitly catch exceptions here, otherwise they would be silently swallowed.
      try
      {
        action(state);
      }
      catch (Exception ex)
      {
        Console.WriteLine("Exception in TaskCounter task:");
        Console.WriteLine(ex.ToString());
        System.Environment.Exit(3);
      }

      // Only signal completion when all tasks have finished.
      if (Interlocked.Decrement(ref activeTaskCount) == 0)
      {
        doneEvent.Set();
      }      
    });
  }


  /// <summary>
  /// Waits for all tasks to complete, throwing an AggregateException if any task failed.
  /// </summary>
  public void Wait()
  {
    Interlocked.Increment(ref waitCallCount);

    double elapsedMs = 0;
    bool measured = false;

    if (!doneEvent.IsSet)
    {
      Stopwatch sw = Stopwatch.StartNew();
      doneEvent.Wait();
      sw.Stop();
      elapsedMs = sw.Elapsed.TotalMilliseconds;
      measured = true;
    }

    if (measured)
    {
      // Update aggregate statistics (without strict synchronization).
      totalWaitTimeMS += elapsedMs;
      sumOfSquaresWaitTimeMS += elapsedMs * elapsedMs;
      if (elapsedMs > maxWaitTimeMS)
      {
        secondMaxWaitTimeMS = maxWaitTimeMS;
        maxWaitTimeMS = elapsedMs;
      }
      else if (elapsedMs > secondMaxWaitTimeMS)
      {
        secondMaxWaitTimeMS = elapsedMs;
      }
    }
  }


  /// <summary>
  /// Resets the task counter to zero, allowing it to be reused.
  /// </summary>
  public void Reset()
  {
    activeTaskCount = 0;
    doneEvent.Set();
  }


  /// <summary>
  /// Returns a string representation of the task counter, including statistics about wait times.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    double avg = waitCallCount > 0 ? totalWaitTimeMS / waitCallCount : 0;
    double variance = waitCallCount > 0
        ? (sumOfSquaresWaitTimeMS - (totalWaitTimeMS * totalWaitTimeMS) / waitCallCount) / waitCallCount
        : 0;
    variance = Math.Max(0, variance);
    double stdDev = Math.Sqrt(variance);

    return $"<TaskCounter tasks={activeTaskCount} waitCalls={waitCallCount} max={maxWaitTimeMS:F1}ms 2nd={secondMaxWaitTimeMS:F1}ms avg={avg:F1}ms std={stdDev:F1}ms tot={totalWaitTimeMS:F1}ms>";
  }


  /// <summary>
  /// Disposes the task counter, releasing any resources.
  /// </summary>
  public void Dispose()
  {
    doneEvent.Dispose();
  }
}
