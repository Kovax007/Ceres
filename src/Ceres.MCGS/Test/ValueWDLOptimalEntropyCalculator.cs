#region Using directives

using System;
using System.Collections.Generic;
using Ceres.Base.Algorithms;

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

namespace Ceres.MCGS.Test;

/// <summary>
/// A calculator for determining the optimal temperature to minimize cross-entropy
/// between target and model WDL (Win, Draw, Loss) values.
/// </summary>
public class ValueWDLOptimalEntropyCalculator
{
  // Internal data structure to store the values
  private readonly List<(float targetW, float targetD, float targetL, float modelW, float modelD, float modelL)> data = [];

  // Method to add WDL values
  public void AddWDL(float targetW, float targetD, float targetL, float modelW, float modelD, float modelL)
  {
    data.Add((targetW, targetD, targetL, modelW, modelD, modelL));
  }


  /// <summary>
  /// Computes the cross-entropy for a given temperature.
  /// </summary>
  /// <param name="temperature"></param>
  /// <returns></returns>
  private float ComputeCrossEntropy(float temperature)
  {
    float crossEntropy = 0;
    const float EPSILON = 1E-7f;

    foreach (var (targetW, targetD, targetL, modelW, modelD, modelL) in data)
    {
      // Apply temperature scaling to the model values
      //        float scaledModelW = (float)Math.Exp(Math.Log(Math.Clamp(modelW, EPSILON, 1 - EPSILON)) / temperature);
      //        float scaledModelD = (float)Math.Exp(Math.Log(Math.Clamp(modelD, EPSILON, 1 - EPSILON)) / temperature);
      //        float scaledModelL = (float)Math.Exp(Math.Log(Math.Clamp(modelL, EPSILON, 1 - EPSILON)) / temperature);

      float scaledModelW = MathF.Pow(Math.Clamp(modelW, EPSILON, 1 - EPSILON), 1 / temperature);
      float scaledModelD = MathF.Pow(Math.Clamp(modelD, EPSILON, 1 - EPSILON), 1 / temperature);
      float scaledModelL = MathF.Pow(Math.Clamp(modelL, EPSILON, 1 - EPSILON), 1 / temperature);

      // Normalize the scaled values
      float sum = scaledModelW + scaledModelD + scaledModelL;
      scaledModelW /= sum;
      scaledModelD /= sum;
      scaledModelL /= sum;

      // Compute cross-entropy for the current data point
      crossEntropy += -(
          targetW * MathF.Log(Math.Clamp(scaledModelW, EPSILON, 1 - EPSILON)) +
          targetD * MathF.Log(Math.Clamp(scaledModelD, EPSILON, 1 - EPSILON)) +
          targetL * MathF.Log(Math.Clamp(scaledModelL, EPSILON, 1 - EPSILON))
      );
    }

    // Average cross-entropy over all data points
    crossEntropy /= data.Count;

    return crossEntropy;
  }


  /// <summary>
  /// Computes the best temperature to minimize cross-entropy.
  /// </summary>
  /// <returns></returns>
  public (float bestTemperature, float bestCrossEntropy) ComputeBestTemperatureToMinimizeCrossEntropy()
  {
    Func<float, float> crossEntropyFunction = ComputeCrossEntropy;
    return Bisection.FindMinimum(crossEntropyFunction, 0.3f, 10f, 0.01f);
  }

  public static double MeanAverageAbsoluteDeviation(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
  {
    if (x.Length != y.Length)
    {
      throw new ArgumentException("x and y must be of the same length");
    }

    double sum = 0;
    for (int i = 0; i < x.Length; i++)
    {
      sum += System.Math.Abs(x[i] - y[i]);
    }

    return sum / x.Length;
  }
}
