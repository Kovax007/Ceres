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

using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Packs two 12‑bit values into a single 24‑bit (3 byte) structure.
  /// 
  /// Speed: reads about 0.2ns and write about 2ns (Ryzen 59x processor).
  /// Atomic updates: not possible.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1, Size = 3)]
  public struct Packed12BitValues
  {
    private const int MaskValue1 = 0x00000FFF;
    private const int MaskValue2 = 0x00FFF000;
    private const int ShiftValue2 = 12;

    private byte _b0;     // least‑significant byte
    private byte _b1;
    private byte _b2;     // most‑significant byte (only lower 4 bits used)


    private int ReadStorage()
    {
      // Recompose a 24‑bit little‑endian integer.
      return _b0 | (_b1 << 8) | (_b2 << 16);
    }

    private void WriteStorage(int value)
    {
      _b0 = (byte)value;
      _b1 = (byte)(value >> 8);
      _b2 = (byte)(value >> 16);
    }

    public ushort Value1
    {
      get => (ushort)(ReadStorage() & MaskValue1);

      //[MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        Debug.Assert(value <= 0x0FFF, "Value1 must be in [0, 4095].");
        int s = ReadStorage();
        s &= ~MaskValue1;         // clear previous bits
        s |= value;
        WriteStorage(s);
      }
    }

    public ushort Value2
    {
      get => (ushort)((ReadStorage() & MaskValue2) >> ShiftValue2);

      set
      {
        Debug.Assert(value <= 0x0FFF, "Value2 must be in [0, 4095].");
        int s = ReadStorage();
        s &= ~MaskValue2;         // clear previous bits
        s |= value << ShiftValue2;
        WriteStorage(s);
      }
    }

    public override string ToString()
    {
      return $"Value1: {Value1}, Value2: {Value2}";
    }
  }
}
