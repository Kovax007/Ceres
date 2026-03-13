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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Storage.Structs;

/// <summary>
/// Represents the index of a node which may be either:
///   - index of existing (materialized) node
///   - index of a non-materialized node (i.e. one that is referenced but not yet created)
/// 
/// Internal representation stores direct as normal ositive values, indirect as their negated indices.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
[Serializable]
public readonly struct NodeIndexPossiblylNonMaterialized : IEquatable<NodeIndexPossiblylNonMaterialized>, IComparable<NodeIndexPossiblylNonMaterialized>
{
  #region Data

  private readonly int index;

  #endregion

  #region Helpers

  public static readonly NodeIndexPossiblylNonMaterialized Null = new NodeIndexPossiblylNonMaterialized(0);

  /// <summary>
  /// Null nodes represent "does not exist"
  /// </summary>
  public bool IsNull => index == 0;

  /// <summary>
  /// Gets the direct index if this is a direct index; otherwise throws.
  /// </summary>
  public NodeIndex IndexDirect
  {
    [DebuggerStepThrough]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get
    {
      Debug.Assert(index >= 0, "IndexDirect called on non-direct index.");
      return new NodeIndex(index);
    }
  }


  /// <summary>
  /// Gets the indirect index if this is an indirect index; otherwise throws.
  /// Returns the absolute value of the negative index.
  /// </summary>
  public NodeIndex IndexIndirect
  {
    [DebuggerStepThrough]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get
    {
      Debug.Assert(index < 0, "Not an indirect index.");
      return new NodeIndex(-index);
    }
  }

  /// <summary>
  /// True if this is a direct index to a materialized node
  /// (the null node is considered materialized).
  /// </summary>
  public bool IsMaterialized => index >= 0;


  #endregion

  #region Constructor/conversion

  [DebuggerStepThrough]
  private NodeIndexPossiblylNonMaterialized(int index) => this.index = index;

  public static NodeIndexPossiblylNonMaterialized CreateDirect(NodeIndex directIndex)
  {
    return new NodeIndexPossiblylNonMaterialized(directIndex.Index);
  }

  public static NodeIndexPossiblylNonMaterialized CreateIndirect(NodeIndex indirectIndex)
  {
    return new NodeIndexPossiblylNonMaterialized(-indirectIndex.Index);
  }

  #endregion

  #region ToString/IEquatable

  public override string ToString()
  {
    if (IsNull)
    {
      return "<NodeIndexPossiblyNonMaterialized [Null]>";
    }
    else if (IsMaterialized)
    {
      return $"<NodeIndexPossiblyNonMaterialized [Direct #{index}]>";
    }
    else
    {
      return $"<NodeIndexPossiblyNonMaterialized [Indirect #{-index}]>";
    }
  }

  public bool Equals(NodeIndexPossiblylNonMaterialized other) => index == other.index;

  public override bool Equals(object obj)
  {
    return obj is NodeIndexPossiblylNonMaterialized other && Equals(other);
  }

  public override int GetHashCode() => index;

  public int CompareTo(NodeIndexPossiblylNonMaterialized other) => index.CompareTo(other.index);

  public static bool operator ==(NodeIndexPossiblylNonMaterialized left, NodeIndexPossiblylNonMaterialized right) => left.Equals(right);

  public static bool operator !=(NodeIndexPossiblylNonMaterialized left, NodeIndexPossiblylNonMaterialized right) => !(left == right);

  #endregion
}
