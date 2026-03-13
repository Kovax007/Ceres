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

namespace Ceres.Base.DataTypes;

/// <summary>
/// A 16‑bit value that packs two logical fields:
/// * 14‑bit unsigned integer  (bits 0‑13)
/// * 2‑bit  unsigned integer  (bits 14‑15)
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 2)]
public struct Packed16As14_2
{
  /// <summary>
  /// Raw 16 bits.
  /// </summary>
  [FieldOffset(0)]
  private ushort _bits;

  private const ushort Mask14Bits = 0x3FFF; // 14 bits (0011 1111 1111 1111)
  private const ushort Mask2Bits = 0xC000; // bits 14‑15 (1100 0000 0000 0000)


  /// <summary>
  /// Returns or sets the 14‑bit unsigned value (0 … 16383).
  /// </summary>
  public ushort Value14BitsUShort
  {
    readonly get
    {
      return (ushort)(_bits & Mask14Bits);
    }
    set
    {
      Debug.Assert(value < 16384);
      _bits = (ushort)((_bits & ~Mask14Bits) | value);
    }
  }


  /// <summary>
  /// Returns or sets the 2‑bit unsigned value (0 … 3).
  /// </summary>
  public byte Value2BitsByte
  {
    readonly get
    {
      return (byte)((_bits >> 14) & 0x03);
    }
    set
    {
      Debug.Assert(value < 4);
      _bits = (ushort)((_bits & ~Mask2Bits) | (ushort)(value << 14));
    }
  }


  /// <summary>
  /// Returns string representation. 
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"{{ Value14BitsUShort={Value14BitsUShort}, " +
           $"Value2BitsByte={Value2BitsByte} }}";
  }
}
