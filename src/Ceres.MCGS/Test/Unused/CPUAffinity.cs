#region Using directives

using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

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

namespace Ceres.MCTS.Test;

internal static class CPUAffinity
{
  public static void LimitProcessToCoreRange(int startLogicalCpuIndex, int count)
  {
    if (startLogicalCpuIndex < 0 || count <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(count), "Start must be >= 0 and count must be > 0.");
    }

    // Find which processor group the start index lives in, and ensure the whole range fits in that group.
    ushort targetGroup;
    int indexWithinGroup;
    int groupSize;
    GetGroupForGlobalIndex(startLogicalCpuIndex, out targetGroup, out indexWithinGroup, out groupSize);

    if (indexWithinGroup + count > groupSize)
    {
      throw new NotSupportedException("Requested core range crosses a processor-group boundary. Use the CPU Sets API version below.");
    }

    // 1) Pin this thread to the target group (threads carry group; processes don't).
    GROUP_AFFINITY threadGa = new()
    {
      Group = targetGroup,
      Mask = BuildMask(indexWithinGroup, count)
    };

    if (!SetThreadGroupAffinity(GetCurrentThread(), ref threadGa, IntPtr.Zero))
    {
      ThrowLastWin32("SetThreadGroupAffinity");
    }

    // 2) Constrain the process to that same mask within the (inherited) group.
    //    SetProcessAffinityMask applies to the process' current group (the scheduler treats the process as group-bound).
    IntPtr hProc = GetCurrentProcess();
    UIntPtr mask = new UIntPtr(threadGa.Mask);

    if (!SetProcessAffinityMask(hProc, mask))
    {
      ThrowLastWin32("SetProcessAffinityMask");
    }
  }

  private static ulong BuildMask(int startIndexInGroup, int count)
  {
    if (startIndexInGroup < 0 || count <= 0 || startIndexInGroup + count > 64)
    {
      throw new ArgumentOutOfRangeException(nameof(count), "Range must fit within 0..63 for a single group.");
    }

    ulong allOnes = (count == 64) ? ulong.MaxValue : ((1UL << count) - 1UL);
    return (allOnes << startIndexInGroup);
  }

  private static void GetGroupForGlobalIndex(int globalIndex, out ushort group, out int indexWithinGroup, out int groupSize)
  {
    ushort groupCount = GetActiveProcessorGroupCount();
    int remaining = globalIndex;

    for (ushort g = 0; g < groupCount; g++)
    {
      int size = (int)GetActiveProcessorCount(g);
      if (remaining < size)
      {
        group = g;
        indexWithinGroup = remaining;
        groupSize = size;
        return;
      }
      remaining -= size;
    }

    throw new ArgumentOutOfRangeException(nameof(globalIndex), "Logical CPU index exceeds system maximum.");
  }

  private static void ThrowLastWin32(string api)
  {
    int err = Marshal.GetLastWin32Error();
    throw new Win32Exception(err, $"{api} failed with error {err}.");
  }

  // -------- Win32 interop --------
  [StructLayout(LayoutKind.Sequential)]
  private struct GROUP_AFFINITY
  {
    public ulong Mask;          // KAFFINITY (64-bit)
    public ushort Group;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public ushort[] Reserved;
  }

  [DllImport("kernel32.dll")]
  private static extern IntPtr GetCurrentProcess();

  [DllImport("kernel32.dll")]
  private static extern IntPtr GetCurrentThread();

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool SetProcessAffinityMask(IntPtr hProcess, UIntPtr dwProcessAffinityMask);

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool SetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY GroupAffinity, IntPtr PreviousGroupAffinity /* optional */);

  [DllImport("kernel32.dll")]
  private static extern ushort GetActiveProcessorGroupCount();

  [DllImport("kernel32.dll")]
  private static extern uint GetActiveProcessorCount(ushort GroupNumber);
}