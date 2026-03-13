#region Using directives

using System;
using Ceres.Base.Benchmarking;

using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.Enumerators;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Storage;


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

namespace Ceres.MCGS.Test
{
  public static class MCGSTestPerformance
  {
    internal static void SpeedTestEnumerateEdges(GameEngineSearchResultCeresMCGS searchResult)
    {
      while (true)
      {
        using (new TimingBlock("xx"))
        {
          RunEdgesTest(searchResult);
        }
      }
    }

    private static void RunEdgesTest(GameEngineSearchResultCeresMCGS searchResult)
    {
      long ex = 0;
      GNode rootNode = searchResult.Engine.Graph.GraphRootNode;
      for (int j = 0; j < 10_000_000; j++)
      {
        foreach (GEdge testChild in rootNode.ChildEdgesExpanded)
        {
          if (true)
          {
            bool foundMyself = false;

            const bool ALLOW_SINGLE_PARENT_FAST_LOOKUP = false;
            if (ALLOW_SINGLE_PARENT_FAST_LOOKUP &&
                testChild.ChildNode.TryGetSingleParentEdge(out GEdge singleParent))
            {
              foundMyself = true;
            }

            else
            {
              ParentEdgesEnumerable parentEdges = testChild.ChildNode.ParentEdges;
              if (true)
              {
                foreach (GEdge testParent in parentEdges)   //BulkMoveWithWriteBarrier size 60 upon calling first time
                                                            // faster               foreach (GNode testParent in testChild.ChildNode.Parents)
                {
                  if (j == 20_000_000) throw new NotImplementedException();
                  foundMyself |= true;// | (testParent.ChildNode == testChild.ChildNode
                                      //&& testParent.ParentNode == testChild.ParentNode);
                  if (foundMyself)
                  {
                    break;
                  }
                }
              }
              else
              {
                foundMyself = true;
              }
            }
            if (!foundMyself)
            {
              throw new Exception("Didn't find myself in parent's children!");
            }
            ex++;
          }

          if (testChild.Q < -22)
          {
            throw new NotImplementedException();
          }
        }
      }
      Console.WriteLine(ex);

    }
  }
}
