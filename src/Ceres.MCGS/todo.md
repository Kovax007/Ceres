Ceres V2 Install Instructions (interim, September 2025)

Prerequisite: Install .NET 9 SDK from https://dotnet.microsoft.com/en-us/download/dotnet/9.0

The following instructions assume Ceres V1 has been pulled from github is up-to-date
and located in:
  C:\dev\Ceres

Installation:
  1. Create a new directory:
       c:\dev\Ceres.MCGS
 
  2. Unpack the files in the CeresV2 ZIP file into this directory so it looks similar to:  
       Directory of C:\dev\Ceres.MCGS
         09/23/2025  08:39 AM    <DIR>          .
         09/23/2025  08:39 AM    <DIR>          ..
         09/23/2025  08:22 AM    <DIR>          .git
         03/17/2025  08:36 PM             6,368 .gitignore
         03/17/2025  08:36 PM            36,421 LICENSE
         03/18/2025  06:17 AM                50 README.md
         09/18/2025  08:50 PM    <DIR>          src

  
  3. Now run Ceres as usual, except referencing Ceres.MCGS.exe from this directory 
     (if necessary, also copy over the Ceres.json into the corresponding directory):
       c:\dev\Ceres.MCGS
     instead of:
      c:\dev\Ceres
     
  
TO DO:
* fix VERSION output (change in Ceres project)
    case "uci":
      const string CERES_VERSION_STR = "2.07";
      // TODO: restore this UCIWriteLine($"id name Ceres {CeresVersion.VersionString}");

* store value2 also in GNodeStruct

* avoid creating 2 executors for every NNEvaluator, share at least weights?

* consider using #IF_ACTION in GMoveInfoStructFields etc. to reduce memory usage

* make all of the low-level data structure classes internal 
  (note: InternalsVisibleTo is already set in project file for test project access)

* add IndexInParent to GEdgeStruct


* SyzygyPly1 seems slow, 10% of runtime, try this position 5k2/6n1/8/6Pp/4KP2/2b5/8/3R4 w - - 0 1
* SyzygyPly1, add option to move generator to only generate captures to make faster
* 
* for performance, eventually make sure coordinator.ParamsSearch.MaxNodes is set as low as possible based on search

* copy MemoryBufferOSBlocked into Ceres.Base

* collect statistics on average number of:
    (1) policy moves per node, and 
    (2) expanded edges per node (make sure not problematic given GEdgeStore.NUM_EDGES_PER_BLOCK)

* move PGNFileEnumerator from CeresTrain to Ceres project

* (UNRELATED) does our deblundering avoid this problem? AnalyzeTerminalBlundersUnexplored farseer

* for efficiency, implement method to take directly EncodedMove (2 bytes) and MakeMove on a MGPosition 
  rather than current conversion to MGMove and then apply

* bug/limitation in prefetcher. at level 4, [0, 0, 0, 1] gets everything but [0, 0, 0, 5] gets very few at level 4

* In Graph.GatherChildInfoViaChildren, the A and UV (and UP) are currently not populated. 
  Improve to do this (only) if needed

* MCGSSelect can be rewritten (ChatGPT) to not use recursion.
  For a search of 500,000 (no parallelism) the benefit could have been as much as 7%.
  Perhaps a better idea is to locate this frame inside the Path class
    // Helper class to hold context for each "recursive" call.
    private class Frame
    {
      public GNode node;
      public int indexInParent;
      public int numTargetVisits;
      public MGPosition mgPos;
      public NodeIndex parentIndex; // From the parent's node index.
    }

* in GraphStore.Validate:
*          // This looks like draw determined by repetition.
          // The edge many have been visited many times, but each time we did not take the value from the child
          // but instead just marked it as a draw directly. So the child N is not related to this edge N.
          // TODO: think about this more. Perhaps the edge needs a field for "was draw by repetition" so we can cleanly track this condition
          //       (what if W were coincidentally 0 for reasons unrelated to draw by repetition?).

* when we create a terminal edge that is a Loss, we'd like to save some LossP that     
  might be slightly less than -1 (distance to mate). But currently there is only W
  and  no LossP/WinP. It needs to start out with N=0 before backup update, so we have
  no way to save this info currently. Find a way.

* In ONNX executor, we have to cast between Span<Half> and Span<FP16>  
  It seems we could do this elegantly with 
    if (typeof(T) == typeof(FP16))
      return MemoryMarshal.Cast<FP16, Half>(s)

* in MiscFields of GNodeStruct, change IsWhite to instead return SideToMove
  (will save some instructions elsewhere)

* perhaps the expensive option of storing Position and Moves with EncodedBatchFlat
  is not needed; now when we evaluate positions in the engine we already have this 
  info handy

* We could have a compaction algorithm
  that uses a bitmap on first 8mm positions to track which recently visited,
  and in the background reorganizes the edges to be in adjacent cache lines

* Ideas for further improving speed:
    - possibly remove MGPosition from the PathVisit, pass this via a field in
      Path called "current leaf position." Maybe same for MoveList, becomes
      "current move list" but have to be careful regarding caching mechanism.

* Use object pool for MGMoveList (the ones that go into last8positions in MGGSPath)

* MCGSCoordinator should have a (careful!) reset operator so we can reuse across searches.

* In GraphStore we have StoresByID[], remove this!

* From HighPerformance library: use Guard, ParallelHelper, and SpanOwner

* If we need to find space for more field(s) in GEdgeStruct, consider:
    - compact P into 1 byte, see my GPT session with "ImportanceWeightedQuantizer"
      (run a test where we simulate this first, see loss of Elo)
    - compact NInFlight1 and NInFlight2 into 3 bytes, see existing class "12bit"

* Add a static flag for "DUMPING_POS_" into NNEvaluator (for debugging).
  Then MCTS/MCGS engines can be run on test positions (possibly size 1)
  to compare if the same raw inputs sent to neural network:
    - the NNEvaluatorBenchmark and NNEvaluatorSet warmup code is disabled
    - the method "IPositionEvaluationBatch EvaluateIntoBuffers"
      looks at this flag and if true, after evaluation does two thing:
      (a) dumps the raw input boards, and (b) dumps the output values (WDL etc)
 
* Tuning opportunity: experiment with values other than 10 or 20. for 
  enums TranspositionHashModeEnum.PosAndStaticHeuristic10Ply.
  Early test results: (1) using 10 rather than none seems to be +5 to +10 Elo @5000 nodes,
                      (2) using 25 rather than none seems to be almost identical results (rarely impacts)

* In the Sygyzy evaluator, when passed MGPosition we first convert to Position then call
  the underlying logic. Someday eliminate the conversion and just pass the MGPosition.
    internal LeafEvaluationResult Lookup(in MGPosition pos) => Lookup(pos.ToPosition);
    internal LeafEvaluationResult Lookup(in Position pos)

* Move MGPositionHashing class to its own source code file.

* Graph has an constructor argument UpdateHashtable that is not much used, maybe unneeded.

* for value head test, UCI should report N (legal moves) not 1 per lepned and Lc0

* line 328 of GameEngineCeresMCGSInProcessNEW is a ResetGame, to be removed
   
* GraphStore, temporarily limit MaxNodes to 50mm!!

* this logic is actually very helpful in V1 (after about 20 ceres.exe)
    "Shutting down, possible infinite process recursion, too many Ceres executables running running"

* try for larger batch size, putting the off-policy ones in the graph but not backup them

* restore prefetch

* AVX512 may now work that crossplatform SIMD is used, but may not be enabled by default (or be faster)

* Unfortunately, Sygyzy tablebases don't make it possible to find the immediate checkmate, 
  hence Ceres plays the capture. It won't affect Elo, but is a little ugly 
  (someday special logic could be added to detect this condition).
  
* Example of above (doesn't choose immediate mate, also verbosemovestats doesn't show all moves):
    setoption name verbosemovestats value true
    setoption name loglivestats value true
    position fen 1k6/R7/5P2/2Qp4/8/8/8/5K2 w - - 1 93
    go nodes 1

* D in W/D/L is approximate. See comments in ResetNodeQ.

* Add visualization back into engine (UCIManagerMCGS).