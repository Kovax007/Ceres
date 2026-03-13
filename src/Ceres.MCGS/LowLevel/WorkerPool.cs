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
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.Base.Threading;

/// <summary>
/// Creates and manages a pool of worker threads that process submitted work items.
/// 
/// TODO: consider this class versus to ParallelItemProcessorWorkerPool
///       (verify correct shutdown logic when work items continue to arrive).
/// </summary>
/// <typeparam name="T"></typeparam>
public sealed class WorkerPool<T> : IDisposable
{

  private readonly BlockingCollection<(Action<T>, T)> pendingWork;
  private readonly ManualResetEventSlim drainedEvent;
  private readonly List<Thread> workers;
  private readonly object workerListLock = new object();
  private readonly CancellationTokenSource cancellationToken;

  private readonly int GrowthIncrement;
  private readonly int MaxThreads;
  private readonly string ThreadNamePrefix;

  private volatile bool shutdownRequested;
  private volatile bool disposed;

  private int pendingWorkCount;              // Enqueued-but-not-finished actions.
  private int activeWorkersCount;            // Currently executing actions.
  private int maxActiveObservedCount;        // High-water mark.
  private int createdThreadCount;       // Threads created.
  private int growthInFlight;           // 0/1 guard to serialize growth.


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="initialThreads"></param>
  /// <param name="growthIncrement"></param>
  /// <param name="maximumThreads"></param>
  /// <param name="threadNamePrefix"></param>
  /// <exception cref="ArgumentOutOfRangeException"></exception>
  public WorkerPool(int initialThreads,
                    int growthIncrement = 2,
                    int? maximumThreads = null,
                    string? threadNamePrefix = null)
  {
    if (initialThreads <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(initialThreads));
    }
    if (growthIncrement <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(growthIncrement));
    }

    GrowthIncrement = growthIncrement;
    MaxThreads = maximumThreads.HasValue ? System.Math.Max(initialThreads, maximumThreads.Value) : int.MaxValue;
    ThreadNamePrefix = threadNamePrefix ?? "WorkerPool";

    // BlockingCollection provides a blocking Take() over a concurrent queue.
    pendingWork = new();
    drainedEvent = new ManualResetEventSlim(true); // no work yet => "drained"
    workers = new List<Thread>();
    cancellationToken = new CancellationTokenSource();
    StartThreads(initialThreads);
  }


  /// <summary>
  /// Enqueue a work item; the delegate receives the worker's per-thread T state.
  /// Supports nested submissions.
  /// </summary>
  public void SubmitWorkItem(Action<T> action, T workItem)
  {
    if (disposed) 
    { 
      throw new ObjectDisposedException(nameof(WorkerPool<T>)); 
    }

    if (Interlocked.Increment(ref pendingWorkCount) == 1)
    {
      drainedEvent.Reset();
    }

    // Unbounded add; returns immediately.
    pendingWork.Add((action, workItem));

    // If we're nearly saturated, consider growing.
    int active = Volatile.Read(ref activeWorkersCount);
    int created = Volatile.Read(ref createdThreadCount);
    if (active >= created - 2)
    {
      TryGrow();
    }
  }


  /// <summary>
  /// Wait until all pending work (including any spawned work) has finished.
  /// Returns the milliseconds spent waiting.
  /// </summary>
  public int WaitAll()
  {
    if (disposed) 
    { 
      throw new ObjectDisposedException(nameof(WorkerPool<T>)); 
    }

    Stopwatch sw = null;

    while (true)
    {
      if (Volatile.Read(ref pendingWorkCount) == 0)
      {
        if (sw == null) 
        { 
          return 0; 
        }

        return (int)sw.ElapsedMilliseconds;
      }

      if (sw == null) 
      { 
        sw = Stopwatch.StartNew(); 
      }
      drainedEvent.Wait();
    }
  }


  /// <summary>
  /// Maximum number of workers observed simultaneously executing user work.
  /// </summary>
  public int MaxConcurrentWorkersObserved
  {
    get { return Volatile.Read(ref maxActiveObservedCount); }
  }

  /// <summary>Current number of dedicated worker threads created for this pool.</summary>
  public int CurrentThreadCount
  {
    get { return Volatile.Read(ref createdThreadCount); }
  }

  public void Dispose()
  {
    Dispose(true);
    GC.SuppressFinalize(this);
  }


  private void StartThreads(int count)
  {
    for (int i = 0; i < count; i++)
    {
      Thread thread = new(WorkerLoop)
      {
        IsBackground = true,
        Name = ThreadNamePrefix + "-" + (Volatile.Read(ref createdThreadCount) + 1).ToString()
      };

      lock (workerListLock)
      {
        workers.Add(thread);
        Interlocked.Increment(ref createdThreadCount);
      }

      thread.Start();
    }
  }


  private void TryGrow()
  {
    if (Volatile.Read(ref shutdownRequested)
     || Interlocked.Exchange(ref growthInFlight, 1) == 1) 
    { 
      return; 
    }

    try
    {
      int created = Volatile.Read(ref createdThreadCount);
      if (created >= MaxThreads) 
      { 
        return;
      }

      int target = System.Math.Min(created + GrowthIncrement, MaxThreads);
      int toCreate = target - created;
      if (toCreate > 0)
      {
        StartThreads(toCreate);
      }
    }
    finally
    {
      Volatile.Write(ref growthInFlight, 0);
    }
  }


  private void WorkerLoop()
  {
    while (true)
    {
      (Action<T>, T)? workItem = null;
      try
      {
        // Blocks until an item arrives or disposal cancels.
        workItem = pendingWork.Take(cancellationToken.Token);
      }
      catch (OperationCanceledException)
      {
        if (shutdownRequested)
        {
          return;
        }
        continue;
      }
      catch (InvalidOperationException)
      {
        // BlockingCollection is marked as CompleteAdding (we don't use that here),
        // but handle defensively.
        if (shutdownRequested)
        {
          return;
        }
        continue;
      }

      int nowActive = Interlocked.Increment(ref activeWorkersCount);
      UpdateMaxActiveObserved(nowActive);

      workItem.Value.Item1(workItem.Value.Item2);
      Interlocked.Decrement(ref activeWorkersCount);

      int left = Interlocked.Decrement(ref pendingWorkCount);
      if (left == 0)
      {
        drainedEvent.Set();
      }
    }
  }


  private void UpdateMaxActiveObserved(int candidate)
  {
    while (true)
    {
      int current = Volatile.Read(ref maxActiveObservedCount);
      if (candidate <= current)
      {
        return;
      }

      if (Interlocked.CompareExchange(ref maxActiveObservedCount, candidate, current) == current)
      {
        return;
      }
    }
  }

  private void Dispose(bool disposing)
  {
    if (!disposing || disposed) 
    { 
      return; 
    }

    disposed = true;
    shutdownRequested = true;

    // Signal cancellation to wake up all blocked workers
    cancellationToken.Cancel();

    // Wait for all worker threads to finish
    List<Thread> threadsToJoin;
    lock (workerListLock)
    {
      threadsToJoin = new List<Thread>(workers);
    }

    foreach (Thread thread in threadsToJoin)
    {
      if (thread.IsAlive)
      {
        thread.Join();
      }
    }

    pendingWork.Dispose();
    drainedEvent.Dispose();
    cancellationToken.Dispose();
  }
}
