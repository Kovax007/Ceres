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
using System.Runtime.CompilerServices;
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Disposable struct that acquires a SpinLockByte on construction.
/// </summary>
public readonly unsafe struct SpinLockByteBlock : IDisposable
{
  /// <summary>
  // Points at the owning SpinLockByte's state field.
  /// </summary>
  private readonly byte* state;


  /// <summary>
  /// Constructor. Acquires lock.
  /// </summary>
  /// <param name="spinLock"></param>
  /// <param name="instanceType"></param>
  public SpinLockByteBlock(ref SpinLockByte spinLock)
  {
    spinLock.Acquire();
    state = (byte*)Unsafe.AsPointer(ref spinLock.state);
  }


  /// <summary>
  /// Releases the lock
  /// </summary>
  public void Dispose() => Volatile.Write(ref *state, (byte)0);  
}
