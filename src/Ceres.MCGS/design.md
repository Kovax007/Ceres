## Repetition Handling

### Goal
Two goals:
  * identify valid transposition nodes where it is safe to merge graphs (for efficiency)
  * correctly identify draw by repetition in the grpah

### Supporting data type - PositionHash96
PositionHash96 is a map from (sequence of MGPosition) --> 96bit has value with two main modes:
* Standalone mode ignores all but the last position in the sequence
* PosAndSequence finds a match of two sequences A,B only if both of the following are true:
  - hash(final(A)) == hash(final(B))
  - set(A) == set(B)
* when in PosAndSequenceMode, the hash is computed by calling Add for each element
  followed by AddFinal with the last element (repeating the use of this element)
* the has described above is also called a "finalized hash" and is what is generally used.
  However during the computation of finalized hash it can happen that a "running hash"
  is maintained internally and used as the basis for ultimately determining "finalized hash" values

### Supporting storage fields
* Each GNodeStruct has a PositionHash96 field (HashStandalone) that always holds the standalone hash of the position
* The MCTSSelect code maintains a running hash of all positions seen since
  last irreversible move. 


### Algorithm (when in GraphEnabled mode)
* MCGSSelect keeps a running hash since last irreversible move.
* When adding a new leaf, computes finalized hash and passes this to Graph.
  The Graph uses this as a lookup/store key for use in the Dictionary. So we will
  only get a match if the hash is in the same equivalence class.
* With this approach, cycle detection is not required. A Cycle will never be found.
* Standalone hash is stored with GNode. Not required! Could be used for cycle detection of paths


