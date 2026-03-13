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
using System.Threading;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Provides a mechanism for acquiring a lock,
/// optionally collecting statistics on wait times.
/// </summary>
public class LockTimed
{
  /// <summary>
  /// If true, allows multiple threads to wait for the lock concurrently and collect statistics.
  /// </summary>
  public readonly bool SupportConcurrentWaiting = false;

  // The internal object for synchronization.
  private readonly object m_lockObject;

  // An extra lock used solely to update statistics.
  private readonly object m_statsLock;

  #region Statistics fields

  // Count of lock acquisitions.
  private long callCount;

  // Sum of wait times (in milliseconds).
  private double totalWait;

  // Sum of squared wait times for stdDev calculation.
  private double totalSquaredWait;

  // Maximum wait time (in milliseconds).
  private long maxWait;

  // Second maximum wait time (in milliseconds).
  private long secondMaxWait;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="supportConcurrentWaiting"></param>
  public LockTimed(bool supportConcurrentWaiting)
  {
    SupportConcurrentWaiting = supportConcurrentWaiting;

    m_lockObject = new object();
    m_statsLock = new object();
    callCount = 0;
    totalWait = 0.0;
    totalSquaredWait = 0.0;
    maxWait = 0;
    secondMaxWait = 0;
    SupportConcurrentWaiting = supportConcurrentWaiting;
  }


  /// <summary>
  /// If the lock is currently held by any thread.
  /// </summary>
  public bool IsEntered => Monitor.IsEntered(m_lockObject);


  /// <summary>
  // Acquire returns a guard that holds the lock.
  // Use it with a using block: using (myTimedLock.Acquire()) { ... }
  /// </summary>
  /// <returns></returns>
  public LockGuard Acquire()
  {
    DateTime start = DateTime.UtcNow;
    Monitor.Enter(m_lockObject);
    DateTime acquired = DateTime.UtcNow;

    // Compute wait time in milliseconds.
    long waitMilliseconds = (long)((acquired - start).TotalMilliseconds);

    void UpdateStats(long waitMilliseconds)
    {
      totalWait += waitMilliseconds;
      totalSquaredWait += waitMilliseconds * waitMilliseconds;

      if (waitMilliseconds > maxWait)
      {
        // When a new maximum is found, the old max becomes the second max.
        secondMaxWait = maxWait;
        maxWait = waitMilliseconds;
      }
      else if (waitMilliseconds > secondMaxWait)
      {
        secondMaxWait = waitMilliseconds;
      }
    }

    if (SupportConcurrentWaiting)
    {
      Interlocked.Increment(ref callCount);

      lock (m_statsLock)
      {
        UpdateStats(waitMilliseconds);
      }
    }
    else
    {
      callCount++;
      UpdateStats(waitMilliseconds);
    }

    return new LockGuard(m_lockObject);
  }


  /// <summary>
  /// Returns a string representation of the lock statistics.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    double avg = 0.0;
    double stdDev = 0.0;
    long callCount = this.callCount; // local copy for consistency

    lock (m_statsLock)
    {
      if (callCount > 0)
      {
        avg = totalWait / callCount;
        double variance = (totalSquaredWait / callCount) - (avg * avg);
        stdDev = (variance > 0.0) ? Math.Sqrt(variance) : 0.0;
      }
    }

    // Format as: "Count: {count}, Avg: {avg} ms, Max: {max} ms, Second Max: {secondMax} ms, StdDev: {stdDev} ms"
    return string.Format($"Count: {callCount,0}, Sum: {totalWait,1:0} ms   Avg: {avg,1:0.00} +/-{stdDev,4:0.00} ms, Max: {maxWait,2:0.00} ms, Second Max: {secondMaxWait,3:0.00} ms, StdDev: {stdDev,4:0.00} ms");
  }


  /// <summary>
  // The disposable guard that releases the lock on Dispose.
  /// </summary>
  public readonly struct LockGuard : IDisposable
  {
    private readonly object m_lock;

    public LockGuard(object theLock)
    {
      m_lock = theLock;
    }

    public void Dispose()
    {
      Monitor.Exit(m_lock);
    }
  }
}


