using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Ceres.MCGS.MCTSNodes.Unused
{

  public static class BloomFilterDemo
  {
    public static void TestBloomFilterLongVersion()
    {
      const int totalSets = 1_000_000;
      const int setSize = 25;

      // We'll pick random values from [0..9999] so duplicates happen more frequently
      const int valueRange = 1_000_000_000;

      long totalBFSignaledDuplicate = 0;  // how many times the BF says "already present"
      long totalRealDuplicates = 0;       // how many times it's actually a duplicate (PP)
      long totalFalsePositives = 0;       // how many times BF is 'positive' but item not in set
      long totalFalseNegatives = 0;       // (should be zero) BF says 'not present' but item is in set

      var random = new Random();

      for (int i = 0; i < totalSets; i++)
      {
        // Create a fresh Bloom Filter for each set
        //SimpleBloomFilterIntrinsics bf = new SimpleBloomFilterIntrinsics();
        BloomFilterSmall bf = new BloomFilterSmall();

        // We'll keep a HashSet of all actual values to detect real duplicates
        HashSet<ulong> groundTruth = new HashSet<ulong>();

        for (int j = 0; j < setSize; j++)
        {
          // Randomly choose a 'ulong' in a smaller range so collisions occur
          ulong val = (ulong)random.Next(valueRange);

          bool bfSaysPresent = bf.ContainsPossibly(val);

          if (true)
          {
            DateTime start = DateTime.Now;
            for (int ix = 0; ix < 1_000_000_000; ix++)
            {
              bfSaysPresent = bf.ContainsPossibly(val);
            }
            Console.WriteLine((DateTime.Now - start).TotalSeconds);
          }
          if (bfSaysPresent)
          {
            // Bloom Filter signaled "we've seen val before"
            totalBFSignaledDuplicate++;

            if (groundTruth.Contains(val))
            {
              // This is actually a real duplicate => "true positive"
              totalRealDuplicates++;
            }
            else
            {
              // Bloom Filter was wrong => "false positive"
              totalFalsePositives++;
            }
          }
          else
          {
            // Bloom Filter says "not present"
            // Check if it actually is in groundTruth => that would be a false negative
            if (groundTruth.Contains(val))
            {
              // We do NOT expect this to happen with a well-configured Bloom Filter
              totalFalseNegatives++;
            }
            else
            {
              // It's genuinely new. Add to Bloom Filter and groundTruth
              bf.Add(val);
              groundTruth.Add(val);
            }
          }
        }
      }

      // Print results
      Console.WriteLine($"Ran {totalSets} random sets of size {setSize}.");
      Console.WriteLine($"Range of values = 0..{valueRange - 1}.");
      Console.WriteLine($"Bloom Filter signaled duplicates (Positive): {totalBFSignaledDuplicate}");
      Console.WriteLine($"  └─ Real duplicates (PP):    {totalRealDuplicates}");
      Console.WriteLine($"  └─ False positives (FP):    {totalFalsePositives}");
      Console.WriteLine($"False negatives (FN):         {totalFalseNegatives}  (expected: 0)");
    }

  }

}
