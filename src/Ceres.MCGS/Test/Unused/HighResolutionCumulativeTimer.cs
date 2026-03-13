#region Using directives

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;


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
  public class HighResolutionCumulativeTimer
  {
#if NOT
    long start = timerSetBatch.Start();
    timerSetBatch.Stop(start);

      public static HighResolutionCumulativeTimer timerSetBatch = new ();
      Console.WriteLine(timerSetBatch.TotalSeconds);
#endif
    // aggregated cycles from QueryPerformanceCounter/RDTSC
    public long TotalTicks;          // read with Volatile.Read

    static float ToSeconds(long ticks) => ticks / (float)Stopwatch.Frequency;
    
    public float TotalSeconds => ToSeconds(TotalTicks);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long Start() => Stopwatch.GetTimestamp();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Stop(long start) => Interlocked.Add(ref TotalTicks, Stopwatch.GetTimestamp() - start);
  }

}
