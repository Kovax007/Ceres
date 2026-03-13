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
using System.Collections.Generic;
using System.Runtime.CompilerServices;


#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Utilities for:
/// (A) Soft/forward-KL "reverse Boltzmann" calibration: Q ≈ τ log π + C(s)
/// (B) Reverse-KL (Grill et al. RPO) mappings between prior μ, improved y, q, and λ
/// (C) τ calibration helpers from per-node data or logs
/// 
/// Value ranges: if your engine uses [-1,1], consider enabling clipping.
/// All logs use natural logarithms (base e).
/// </summary>
public static class BoltzmannValueCalibrator
{
  // ----------------------------- Types & options -----------------------------

  public sealed record StateLog(double[] Policy, double[] Q);

  public enum NodeWeighting { Uniform, PriorPi, ImprovedPolicy }

  // ----------------------------- Public API (double) -----------------------------

  /// <summary>
  /// Forward-KL/entropy: calibrated child Q from parent value anchor.
  /// Q_i = v_parent + τ ( log π_i + H(π) ), ensuring E_π[Q]=v_parent.
  /// </summary>
  public static void ComputeQFromPolicy_MatchParentValue(
      ReadOnlySpan<double> pi,
      double parentValue,
      double tau,
      Span<double> qOut,
      bool renormalizeIfNeeded = true,
      double epsilon = 1e-12,
      bool clipToRange = false,
      double clipMin = -1.0,
      double clipMax = 1.0)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    double sum = Sum(pi);
    if (renormalizeIfNeeded && (sum <= 0 || Math.Abs(sum - 1.0) > 1e-9))
    {
      ScaleInPlace(pi, 1.0 / Math.Max(sum, epsilon), out var piNorm);
      ComputeQFromPolicy_MatchParentValue(piNorm, parentValue, tau, qOut,
                                          renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
      return;
    }

    double H = 0.0;
    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      H -= p * Math.Log(p);
    }

    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      double qi = parentValue + tau * (Math.Log(p) + H);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = qi;
    }
  }

  public static void ComputeQFromPolicy_AnchorChild(
    ReadOnlySpan<double> pi,
    int anchorIndex,
    double anchorQ,
    double tau,
    Span<double> qOut,
    bool renormalizeIfNeeded = true,
    double epsilon = 1e-12,
    bool clipToRange = false,
    double clipMin = -1.0,
    double clipMax = 1.0)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    if ((uint)anchorIndex >= (uint)pi.Length)
      throw new ArgumentOutOfRangeException(nameof(anchorIndex));

    if (renormalizeIfNeeded)
    {
      double sum = Sum(pi);
      if (sum <= 0 || Math.Abs(sum - 1.0) > 1e-9)
      {
        ScaleInPlace(pi, 1.0 / Math.Max(sum, epsilon), out var piNorm);
        ComputeQFromPolicy_AnchorChild(piNorm, anchorIndex, anchorQ, tau, qOut,
                                       renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
        return;
      }
    }

    // Scalar log for anchor stays unchanged
    double logAnchor = Math.Log(ClampProb(pi[anchorIndex], epsilon));

    int n = pi.Length;
    if (n == 0) return;

    // Work buffers for clamped pi and log(pi)
    double[] pBuf = System.Buffers.ArrayPool<double>.Shared.Rent(n);
    double[] logBuf = System.Buffers.ArrayPool<double>.Shared.Rent(n);

    try
    {
      var pSpan = new Span<double>(pBuf, 0, n);
      var logSpan = new Span<double>(logBuf, 0, n);

      // Copy and clamp like ClampProb: p <= eps => eps; p >= 1 => 1 - 1e-16
      const double oneMinusTiny = 1.0 - 1e-16;
      for (int i = 0; i < n; i++)
      {
        double v = pi[i];
        if (v <= epsilon) v = epsilon;
        else if (v >= 1.0) v = oneMinusTiny;
        pSpan[i] = v;
      }

      // Vectorized log over the span
      System.Numerics.Tensors.TensorPrimitives.Log(pSpan, logSpan);

      // Vectorized affine transform and optional clipping
      int width = System.Numerics.Vector<double>.Count;
      var vTau = new System.Numerics.Vector<double>(tau);
      var vAnchorQ = new System.Numerics.Vector<double>(anchorQ);
      var vLogAnchor = new System.Numerics.Vector<double>(logAnchor);

      int iVec = 0;
      if (!clipToRange)
      {
        for (; iVec <= n - width; iVec += width)
        {
          var vLog = new System.Numerics.Vector<double>(logSpan.Slice(iVec, width));
          var v = vAnchorQ + vTau * (vLog - vLogAnchor);
          v.CopyTo(qOut.Slice(iVec, width));
        }
        for (; iVec < n; iVec++)
        {
          qOut[iVec] = anchorQ + tau * (logSpan[iVec] - logAnchor);
        }
      }
      else
      {
        var vMin = new System.Numerics.Vector<double>(clipMin);
        var vMax = new System.Numerics.Vector<double>(clipMax);

        for (; iVec <= n - width; iVec += width)
        {
          var vLog = new System.Numerics.Vector<double>(logSpan.Slice(iVec, width));
          var v = vAnchorQ + vTau * (vLog - vLogAnchor);
          v = System.Numerics.Vector.Min(System.Numerics.Vector.Max(v, vMin), vMax);
          v.CopyTo(qOut.Slice(iVec, width));
        }
        for (; iVec < n; iVec++)
        {
          double qi = anchorQ + tau * (logSpan[iVec] - logAnchor);
          if (qi < clipMin) qi = clipMin;
          else if (qi > clipMax) qi = clipMax;
          qOut[iVec] = qi;
        }
      }
    }
    finally
    {
      System.Buffers.ArrayPool<double>.Shared.Return(pBuf);
      System.Buffers.ArrayPool<double>.Shared.Return(logBuf);
    }
  }
  /// <summary>
  /// Forward-KL/entropy: calibrated child Q from anchoring a single child a*.
  /// Q_i = Q_* + τ (log π_i - log π_*).
  /// </summary>
  public static void ComputeQFromPolicy_AnchorChildSLOW(
      ReadOnlySpan<double> pi,
      int anchorIndex,
      double anchorQ,
      double tau,
      Span<double> qOut,
      bool renormalizeIfNeeded = true,
      double epsilon = 1e-12,
      bool clipToRange = false,
      double clipMin = -1.0,
      double clipMax = 1.0)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    if ((uint)anchorIndex >= (uint)pi.Length)
      throw new ArgumentOutOfRangeException(nameof(anchorIndex));

    if (renormalizeIfNeeded)
    {
      double sum = Sum(pi);
      if (sum <= 0 || Math.Abs(sum - 1.0) > 1e-9)
      {
        ScaleInPlace(pi, 1.0 / Math.Max(sum, epsilon), out var piNorm);
        ComputeQFromPolicy_AnchorChild(piNorm, anchorIndex, anchorQ, tau, qOut,
                                       renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
        return;
      }

    }
    double logAnchor = Math.Log(ClampProb(pi[anchorIndex], epsilon));
    for (int i = 0; i < pi.Length; i++)
    {
      double logp = Math.Log(ClampProb(pi[i], epsilon));
      double qi = anchorQ + tau * (logp - logAnchor);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = qi;
    }
  }

  /// <summary>
  /// Reverse-KL (Grill et al. RPO): derive per-child Q from prior μ, improved y, parent value v, and λ.
  /// Formula: Q(a) = v + λ ( 1 - μ(a) / y(a) ).
  /// Typically y is the visit distribution (normalized) or the closed-form optimizer.
  /// </summary>
  public static void ComputeQFromPriorAndImprovedPolicy_ReverseKL(
      ReadOnlySpan<double> mu,               // prior policy π_θ
      ReadOnlySpan<double> improved,         // improved policy y (e.g., normalized visits)
      double parentValue,                    // v(s)
      double lambda,                         // λ_N
      Span<double> qOut,
      bool renormalizeIfNeeded = true,
      double epsilon = 1e-12,
      bool clipToRange = false,
      double clipMin = -1.0,
      double clipMax = 1.0)
  {
    if (mu.Length != improved.Length || mu.Length != qOut.Length)
      throw new ArgumentException("All spans must have identical length.");
    if (!(lambda >= 0.0)) throw new ArgumentOutOfRangeException(nameof(lambda), "λ must be ≥ 0.");
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = mu.Length;

    double sMu = Sum(mu);
    double sY = Sum(improved);
    if (renormalizeIfNeeded)
    {
      if (sMu <= 0) throw new ArgumentException("Prior μ has non-positive sum.");
      if (sY <= 0) throw new ArgumentException("Improved policy y has non-positive sum.");
    }
    double invMu = renormalizeIfNeeded ? 1.0 / sMu : 1.0;
    double invY = renormalizeIfNeeded ? 1.0 / sY : 1.0;

    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(mu[i] * invMu, epsilon);
      double ya = ClampProb(improved[i] * invY, epsilon);
      double qi = parentValue + lambda * (1.0 - p / ya);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = qi;
    }
  }

  /// <summary>
  /// Reverse-KL optimizer: given q, prior μ, and λ, compute y* = argmax_y { q^T y - λ KL[μ, y] }.
  /// Closed form: y*(a) = λ μ(a) / (α - q(a)), with α chosen so Σ y*(a)=1 and α > max_a q(a).
  /// Uses safe bisection for α.
  /// </summary>
  public static void ComputeImprovedPolicy_ReverseKL(
      ReadOnlySpan<double> q,
      ReadOnlySpan<double> mu,
      double lambda,
      Span<double> yOut,
      bool renormalizeMuIfNeeded = true,
      double epsilon = 1e-12,
      int maxBisectionIters = 60)
  {
    if (q.Length != mu.Length || q.Length != yOut.Length)
      throw new ArgumentException("q, μ, and yOut must have the same length.");
    if (!(lambda > 0)) throw new ArgumentOutOfRangeException(nameof(lambda), "λ must be > 0.");
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = q.Length;

    // Normalize μ if requested
    double sumMu = Sum(mu);
    if (renormalizeMuIfNeeded)
    {
      if (sumMu <= 0) throw new ArgumentException("μ has non-positive sum.");
    }
    double invMu = renormalizeMuIfNeeded ? 1.0 / sumMu : 1.0;

    // α must be strictly greater than max q(a) to keep denominators positive
    double qMax = q[0];
    for (int i = 1; i < n; i++) if (q[i] > qMax) qMax = q[i];

    // Bracket α: lower just above qMax; upper grows until Σ y(α) <= 1
    double lower = qMax + 1e-12;
    double upper = lower * 2 + 1.0; // initial guess
    for (int guard = 0; guard < 60; guard++)
    {
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = upper - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY <= 1.0) break;
      upper = upper * 2 + 1.0;
    }

    // Bisection to solve Σ y(α) = 1
    for (int it = 0; it < maxBisectionIters; it++)
    {
      double mid = 0.5 * (lower + upper);
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = mid - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY > 1.0) lower = mid; else upper = mid;
    }
    double alpha = 0.5 * (lower + upper);

    // Compute y*(a)
    double sumYFinal = 0.0;
    for (int i = 0; i < n; i++)
    {
      double denom = alpha - q[i];
      double mui = ClampProb(mu[i] * invMu, epsilon);
      double yi = (lambda * mui) / denom;
      yOut[i] = yi;
      sumYFinal += yi;
    }
    // Normalize to sum exactly 1 (small numerical drift)
    if (sumYFinal > 0)
    {
      double inv = 1.0 / sumYFinal;
      for (int i = 0; i < n; i++) yOut[i] *= inv;
    }
  }

  /// <summary>
  /// τ fit at a single node: regress Q on log π with a per-node intercept removed.
  /// Returns slope α ≈ τ. Choose weighting scheme: Uniform, PriorPi, or ImprovedPolicy.
  /// </summary>
  public static double FitTauAtNode(
      ReadOnlySpan<double> pi,
      ReadOnlySpan<double> q,
      NodeWeighting weighting = NodeWeighting.PriorPi,
      ReadOnlySpan<double> improvedPolicy = default,
      double epsilon = 1e-12)
  {
    if (pi.Length != q.Length) throw new ArgumentException("pi and q must have the same length.");
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = pi.Length;

    // Normalize π and (optionally) y
    double sPi = Sum(pi);
    if (sPi <= 0) throw new ArgumentException("π has non-positive sum.");
    double invPi = 1.0 / sPi;

    double invY = 1.0;
    bool useY = weighting == NodeWeighting.ImprovedPolicy;
    if (useY)
    {
      if (improvedPolicy.Length != n) throw new ArgumentException("improvedPolicy length mismatch.");
      double sY = Sum(improvedPolicy);
      if (sY <= 0) throw new ArgumentException("Improved policy has non-positive sum.");
      invY = 1.0 / sY;
    }

    // Weighted means of x = log π, y = Q
    double wSum = 0.0, xBar = 0.0, yBar = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(pi[i] * invPi, epsilon);
      double w = weighting switch
      {
        NodeWeighting.Uniform => 1.0,
        NodeWeighting.PriorPi => p,
        NodeWeighting.ImprovedPolicy => ClampProb(improvedPolicy[i] * invY, epsilon),
        _ => 1.0
      };
      double x = Math.Log(p);
      double yy = q[i];
      wSum += w; xBar += w * x; yBar += w * yy;
    }
    if (wSum <= 0) throw new InvalidOperationException("No positive weights.");
    xBar /= wSum; yBar /= wSum;

    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(pi[i] * invPi, epsilon);
      double w = weighting switch
      {
        NodeWeighting.Uniform => 1.0,
        NodeWeighting.PriorPi => p,
        NodeWeighting.ImprovedPolicy => ClampProb(improvedPolicy[i] * invY, epsilon),
        _ => 1.0
      };
      double x = Math.Log(p);
      double dx = x - xBar;
      double dy = q[i] - yBar;
      sxx += w * dx * dx;
      sxy += w * dx * dy;
    }
    if (sxx <= 0) throw new InvalidOperationException("Insufficient variance in log π.");
    return sxy / sxx; // α ≈ τ
  }

  /// <summary>
  /// τ fit at a node using RPO-implied Q as the target: 
  /// First compute Q_RPO from (μ, y, v, λ), then fit τ in Q_RPO ≈ τ log π + C.
  /// </summary>
  public static double FitTauAtNodeFromRPO(
      ReadOnlySpan<double> mu,
      ReadOnlySpan<double> improved,
      double parentValue,
      double lambda,
      NodeWeighting weighting = NodeWeighting.PriorPi,
      double epsilon = 1e-12)
  {
    int n = mu.Length;
    if (improved.Length != n) throw new ArgumentException("mu and improved length mismatch.");
    var qTmp = new double[n];
    ComputeQFromPriorAndImprovedPolicy_ReverseKL(mu, improved, parentValue, lambda, qTmp,
                                                 renormalizeIfNeeded: true, epsilon: epsilon,
                                                 clipToRange: false);
    return FitTauAtNode(mu, qTmp, weighting, improved, epsilon);
  }

  /// <summary>
  /// Fit τ globally from many nodes.
  /// Q ≈ τ log π + C(s), with within-node demeaning; optionally weight by π.
  /// </summary>
  public static double FitTauFromLogs(IEnumerable<StateLog> logs, bool weightByPi = true, double epsilon = 1e-12)
  {
    if (logs is null) throw new ArgumentNullException(nameof(logs));

    double numer = 0.0, denom = 0.0;

    foreach (var state in logs)
    {
      var pi = state.Policy;
      var q = state.Q;
      if (pi == null || q == null) continue;
      int n = Math.Min(pi.Length, q.Length);
      if (n <= 1) continue;

      double sum = 0.0;
      for (int i = 0; i < n; i++) sum += pi[i];
      if (sum <= 0) continue;
      double invSum = 1.0 / sum;

      double wSum = 0.0, xBar = 0.0, yBar = 0.0;
      for (int i = 0; i < n; i++)
      {
        double p = ClampProb(pi[i] * invSum, epsilon);
        double w = weightByPi ? p : 1.0;
        double x = Math.Log(p);
        double y = q[i];
        wSum += w; xBar += w * x; yBar += w * y;
      }
      if (wSum <= 0) continue;
      xBar /= wSum; yBar /= wSum;

      double sxx = 0.0, sxy = 0.0;
      for (int i = 0; i < n; i++)
      {
        double p = ClampProb(pi[i] * invSum, epsilon);
        double w = weightByPi ? p : 1.0;
        double x = Math.Log(p);
        double y = q[i];
        double dx = x - xBar;
        double dy = y - yBar;
        sxx += w * dx * dx;
        sxy += w * dx * dy;
      }
      if (sxx <= 0) continue;

      numer += sxy;
      denom += sxx;
    }

    if (denom <= 0) throw new InvalidOperationException("Insufficient variance in log π to fit τ.");
    return numer / denom;
  }

  /// <summary>
  /// Compute λ_N from visit counts according to the Grill et al.-style scaling:
  /// λ_N = c * sqrt(N_total) / ( |A| + N_total ).
  /// Tune c to your domain; larger c strengthens regularization.
  /// </summary>
  public static double ComputeLambdaFromCounts(ReadOnlySpan<int> visitCounts, double c = 1.0)
  {
    long N = 0; for (int i = 0; i < visitCounts.Length; i++) N += Math.Max(visitCounts[i], 0);
    int A = Math.Max(visitCounts.Length, 1);
    return c * Math.Sqrt(Math.Max(0L, N)) / (A + Math.Max(0L, N));
  }

  /// <summary>
  /// Normalize integer visit counts into a probability distribution (optionally with a tiny additive prior).
  /// </summary>
  public static void NormalizeCounts(ReadOnlySpan<int> counts, Span<double> yOut, double additivePrior = 0.0)
  {
    if (counts.Length != yOut.Length) throw new ArgumentException("counts and yOut must have same length.");
    double sum = 0.0;
    for (int i = 0; i < counts.Length; i++)
    {
      double val = Math.Max(0, counts[i]) + additivePrior;
      yOut[i] = val; sum += val;
    }
    if (sum <= 0) { double inv = 1.0 / counts.Length; for (int i = 0; i < counts.Length; i++) yOut[i] = inv; return; }
    double invSum = 1.0 / sum;
    for (int i = 0; i < counts.Length; i++) yOut[i] *= invSum;
  }

  // ----------------------------- Public API (float) -----------------------------

  public static void ComputeQFromPolicy_MatchParentValue(
      ReadOnlySpan<float> pi,
      float parentValue,
      float tau,
      Span<float> qOut,
      bool renormalizeIfNeeded = true,
      float epsilon = 1e-12f,
      bool clipToRange = false,
      float clipMin = -1f,
      float clipMax = 1f)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    float sum = Sum(pi);
    if (renormalizeIfNeeded && (sum <= 0 || Math.Abs(sum - 1f) > 1e-7f))
    {
      ScaleInPlace(pi, 1f / Math.Max(sum, epsilon), out var piNorm);
      ComputeQFromPolicy_MatchParentValue(piNorm, parentValue, tau, qOut,
                                          renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
      return;
    }

    double H = 0.0;
    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      H -= p * Math.Log(p);
    }

    for (int i = 0; i < pi.Length; i++)
    {
      double p = ClampProb(pi[i], epsilon);
      double qi = parentValue + tau * (Math.Log(p) + H);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = (float)qi;
    }
  }

  public static void ComputeQFromPolicy_AnchorChild(
      ReadOnlySpan<float> pi,
      int anchorIndex,
      float anchorQ,
      float tau,
      Span<float> qOut,
      bool renormalizeIfNeeded = true,
      float epsilon = 1e-12f,
      bool clipToRange = false,
      float clipMin = -1f,
      float clipMax = 1f)
  {
    ThrowIfInvalidArgs(pi, qOut, tau, epsilon);
    if ((uint)anchorIndex >= (uint)pi.Length)
      throw new ArgumentOutOfRangeException(nameof(anchorIndex));

    float sum = Sum(pi);
    if (renormalizeIfNeeded && (sum <= 0 || Math.Abs(sum - 1f) > 1e-7f))
    {
      ScaleInPlace(pi, 1f / Math.Max(sum, epsilon), out var piNorm);
      ComputeQFromPolicy_AnchorChild(piNorm, anchorIndex, anchorQ, tau, qOut,
                                     renormalizeIfNeeded: false, epsilon, clipToRange, clipMin, clipMax);
      return;
    }

    double logAnchor = Math.Log(ClampProb(pi[anchorIndex], epsilon));
    for (int i = 0; i < pi.Length; i++)
    {
      double logp = Math.Log(ClampProb(pi[i], epsilon));
      double qi = anchorQ + tau * (logp - logAnchor);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = (float)qi;
    }
  }

  public static void ComputeQFromPriorAndImprovedPolicy_ReverseKL(
      ReadOnlySpan<float> mu,
      ReadOnlySpan<float> improved,
      float parentValue,
      float lambda,
      Span<float> qOut,
      bool renormalizeIfNeeded = true,
      float epsilon = 1e-12f,
      bool clipToRange = false,
      float clipMin = -1f,
      float clipMax = 1f)
  {
    if (mu.Length != improved.Length || mu.Length != qOut.Length)
      throw new ArgumentException("All spans must have identical length.");
    if (!(lambda >= 0f)) throw new ArgumentOutOfRangeException(nameof(lambda));
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = mu.Length;

    float sMu = Sum(mu);
    float sY = Sum(improved);
    if (renormalizeIfNeeded)
    {
      if (sMu <= 0) throw new ArgumentException("Prior μ has non-positive sum.");
      if (sY <= 0) throw new ArgumentException("Improved policy y has non-positive sum.");
    }
    float invMu = renormalizeIfNeeded ? 1f / sMu : 1f;
    float invY = renormalizeIfNeeded ? 1f / sY : 1f;

    for (int i = 0; i < n; i++)
    {
      double p = ClampProb(mu[i] * invMu, epsilon);
      double ya = ClampProb(improved[i] * invY, epsilon);
      double qi = parentValue + lambda * (1.0 - p / ya);
      if (clipToRange) qi = Math.Max(clipMin, Math.Min(clipMax, qi));
      qOut[i] = (float)qi;
    }
  }

  public static void ComputeImprovedPolicy_ReverseKL(
      ReadOnlySpan<float> q,
      ReadOnlySpan<float> mu,
      float lambda,
      Span<float> yOut,
      bool renormalizeMuIfNeeded = true,
      float epsilon = 1e-12f,
      int maxBisectionIters = 60)
  {
    if (q.Length != mu.Length || q.Length != yOut.Length)
      throw new ArgumentException("q, μ, and yOut must have the same length.");
    if (!(lambda > 0)) throw new ArgumentOutOfRangeException(nameof(lambda));
    if (!(epsilon > 0 && epsilon < 1)) throw new ArgumentOutOfRangeException(nameof(epsilon));

    int n = q.Length;

    float sumMu = Sum(mu);
    if (renormalizeMuIfNeeded)
    {
      if (sumMu <= 0) throw new ArgumentException("μ has non-positive sum.");
    }
    float invMu = renormalizeMuIfNeeded ? 1f / sumMu : 1f;

    float qMax = q[0];
    for (int i = 1; i < n; i++) if (q[i] > qMax) qMax = q[i];

    double lower = qMax + 1e-12;
    double upper = lower * 2 + 1.0;
    for (int guard = 0; guard < 60; guard++)
    {
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = upper - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY <= 1.0) break;
      upper = upper * 2 + 1.0;
    }

    for (int it = 0; it < maxBisectionIters; it++)
    {
      double mid = 0.5 * (lower + upper);
      double sumY = 0.0;
      for (int i = 0; i < n; i++)
      {
        double denom = mid - q[i];
        if (denom <= 0) { sumY = double.PositiveInfinity; break; }
        double mui = ClampProb(mu[i] * invMu, epsilon);
        sumY += (lambda * mui) / denom;
      }
      if (sumY > 1.0) lower = mid; else upper = mid;
    }
    double alpha = 0.5 * (lower + upper);

    double sumYFinal = 0.0;
    for (int i = 0; i < n; i++)
    {
      double denom = alpha - q[i];
      double mui = ClampProb(mu[i] * invMu, epsilon);
      double yi = (lambda * mui) / denom;
      yOut[i] = (float)yi;
      sumYFinal += yi;
    }
    if (sumYFinal > 0)
    {
      double inv = 1.0 / sumYFinal;
      for (int i = 0; i < n; i++) yOut[i] = (float)(yOut[i] * inv);
    }
  }

  public static double ComputeLambdaFromCounts(ReadOnlySpan<float> visitCounts, double c = 1.0)
  {
    double N = 0.0; for (int i = 0; i < visitCounts.Length; i++) N += Math.Max(visitCounts[i], 0.0f);
    int A = Math.Max(visitCounts.Length, 1);
    return c * Math.Sqrt(Math.Max(0.0, N)) / (A + Math.Max(0.0, N));
  }

  public static void NormalizeCounts(ReadOnlySpan<int> counts, Span<float> yOut, float additivePrior = 0.0f)
  {
    if (counts.Length != yOut.Length) throw new ArgumentException("counts and yOut must have same length.");
    double sum = 0.0;
    for (int i = 0; i < counts.Length; i++)
    {
      double val = Math.Max(0, counts[i]) + additivePrior;
      yOut[i] = (float)val; sum += val;
    }
    if (sum <= 0) { float inv = 1f / counts.Length; for (int i = 0; i < counts.Length; i++) yOut[i] = inv; return; }
    float invSum = (float)(1.0 / sum);
    for (int i = 0; i < counts.Length; i++) yOut[i] *= invSum;
  }

  // ----------------------------- Helpers & guards -----------------------------

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double ClampProb(double p, double eps) => p <= eps ? eps : (p >= 1.0 ? (1.0 - 1e-16) : p);

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double Sum(ReadOnlySpan<double> x)
  {
    double s = 0.0; for (int i = 0; i < x.Length; i++) s += x[i]; return s;
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static float Sum(ReadOnlySpan<float> x)
  {
    float s = 0f; for (int i = 0; i < x.Length; i++) s += x[i]; return s;
  }

  private static void ScaleInPlace(ReadOnlySpan<double> x, double scale, out double[] y)
  {
    y = new double[x.Length];
    for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale;
  }

  private static void ScaleInPlace(ReadOnlySpan<float> x, float scale, out float[] y)
  {
    y = new float[x.Length];
    for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale;
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void ThrowIfInvalidArgs(ReadOnlySpan<double> pi, Span<double> qOut, double tau, double eps)
  {
    if (pi.Length != qOut.Length) throw new ArgumentException("pi and qOut must have the same length.");
    if (!(tau > 0.0)) throw new ArgumentOutOfRangeException(nameof(tau), "τ must be > 0.");
    if (!(eps > 0.0 && eps < 1.0)) throw new ArgumentOutOfRangeException(nameof(eps), "epsilon must be in (0,1).");
  }

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static void ThrowIfInvalidArgs(ReadOnlySpan<float> pi, Span<float> qOut, float tau, float eps)
  {
    if (pi.Length != qOut.Length) throw new ArgumentException("pi and qOut must have the same length.");
    if (!(tau > 0f)) throw new ArgumentOutOfRangeException(nameof(tau), "τ must be > 0.");
    if (!(eps > 0f && eps < 1f)) throw new ArgumentOutOfRangeException(nameof(eps), "epsilon must be in (0,1).");
  }


  public static void TestBoltzmann()
  {
    //					public static double FitTauFromLogs(IEnumerable<StateLog> logs, bool weightByPi = true, double epsilon = 1e-12)

    double[] newPol = new double[2];
    double[] q = [0.664, 0.600];
    double[] pi = [0.769, 0.231];
    ComputeImprovedPolicy_ReverseKL(q: q, mu: pi, lambda: 0.064, newPol); // ==> [0.55, 0.45]

    //ComputeImprovedPolicy_ReverseKL([0.2, 0.1], [0.3, 0.7], 0.3, newPol); // ==> [0.37, 0.63]

    // Example 1: derive child Q's by matching the parent value
    //double[] pi = [0.769, 0.231]; /* policy head at state s, length = number of legal moves */
    //double vParent = 0.20;/* value head at s, e.g., in [-1,1] */

    double tau = 0.05; // start with a guess, or fit from logs (below)
    double parentValue = q[0];
    BoltzmannValueCalibrator.ComputeQFromPolicy_MatchParentValue(pi, parentValue, tau, q,
                                                                renormalizeIfNeeded: true, epsilon: 1e-12,
                                                                clipToRange: true, clipMin: -1, clipMax: 1);

    // q[i] is a calibrated estimate you can use to initialize unexpanded edges.

    // Example 2: anchor to a trusted best child
    int bestIdx = 0;/* index chosen by your move ordering or partial search */
    double qBest = q[0];/* current MCTS estimate for that child */
    var q2 = new double[pi.Length];
    BoltzmannValueCalibrator.ComputeQFromPolicy_AnchorChild(pi, bestIdx, qBest, tau, q2,
                                                            renormalizeIfNeeded: true, epsilon: 1e-12,
                                                            clipToRange: true, clipMin: -1, clipMax: 1);

    // Example 3: fit τ from your own search logs
    var logs = new List<BoltzmannValueCalibrator.StateLog>();
    // Populate logs with (policy, per-child Q) snapshots at roots (or selected nodes).
    // Q here should be your current best estimates from search for children of s.

    double fittedTau = BoltzmannValueCalibrator.FitTauFromLogs(logs, weightByPi: true, epsilon: 1e-12);
    // Now reuse fittedTau in the calls above.
  }
}

#if NOT
      // Compute best fit for tau from some of the nodes in the graph.
      if (COUNT++ % 4777 == 32 && graph.GraphRootNode.N > 220)// && graph.GraphRootNode.Index.Index % 77 ==43)
      {
        List<BoltzmannValueCalibrator.StateLog> logs = new();
        for (int i = 1; i < graph.GraphRootNode.N; i++)
        {
          GNode testNode = graph[i];
//          using (new SpinLockByteBlock(ref testNode.NodeRef.LockRef))
          {
            if (testNode.N > 20)
            {
              double[] q = new double[testNode.NumEdgesExpanded];
              double[] policy = new double[testNode.NumEdgesExpanded];
              for (int j = 0; j < testNode.NumEdgesExpanded; j++)
              {
                if (testNode.ChildEdgeAtIndex(j).Type != GEdgeStruct.EdgeType.ChildEdge)
                {
                  break;
                }
                //              q[j] = testNode.ChildEdgeAtIndex(j).Q;
                //                q[j] = testNode.Q;
                q[j] = -testNode.ChildEdgeAtIndex(j).ChildNode.V;// .Q;
                policy[j] = testNode.ChildEdgeAtIndex(j).P;
              }

              logs.Add(new(policy, q));
            }
          }
        }

        double taux = BoltzmannValueCalibrator.FitTauFromLogs(logs);//, bool weightByPi = true
        if (!double.IsNaN(taux)) Console.WriteLine(logs.Count + " " + taux);        
      }
#endif

