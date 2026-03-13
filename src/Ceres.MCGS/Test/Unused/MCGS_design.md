-- MCGS
# MCGS Overview

GVisitTo records an MCGS path passing from a parent position to child position.

GVisitFrom records

## Raw Structs

|Name|Description|#Fields|Size|# per node
|----|-----------|-------|----|-----
|GNodeStruct|Node basic data|about 30|64 bytes|1
|GMoveInfoStruct|per move data (move/policy%/actionV/actionU)|4|8 bytes|0 to MAX_MOVES
|GChildStruct|Visits to a node|1|32|0 to many
|VisitFromStruct|Visits from a node|1|8|0 to many
|AllStates|Prior position states|1|64|1

A single GNodeStruct exists for each node in the tree.

Each GNodeStruct has an array of GMoveInfoStructs, one for each possible move from the node
containing the raw policy information:
* move 
* policy probability
* action value
* action uncertainty.

* Each GNodeChild contains visit information including:
* 
* 
* 
* 

GraphStore
* NodesStore
* MoveInfoStore
* VisitsToStore
* VisitsFromStore

* 
Graph
  Nodes
  MovesInfo
  VisitsTo
  VisitsFrom

Node
  Ref --> NodeStruct

  ChildAccessor this[int childIndex]
    MoveInfo
    VisitsTo
    VisitsFrom
    Node
    Ref



## Graph Update Primitives
Node root = graph.Initialize();
Span<MoveInfoStruct> root.AllocateMoveInfos(int numMoves)

int childIndex = 0;
Node child = root.AddChild(childIndex);
VisitsToStruct rootVisits = root[childIndex].VisitsTo;
rootVisits[0].N = 1;


NodeStruct  Nod

| Struct Name   | Description   | Cool  |
| ------------- |:-------------:| -----:|
| NodeStruct    | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


- Structs
- Stores
- Node classes
- MCGS classes


# Wrapper classes

  Inside wrapper classes, we use pointer to the underlying structs in the graph data structures.

  The use pointers here requires an unsafe context in C#.

  An alternative design would be to use a `ref struct`, which enables safe stack-only references without unsafe code.
  However, `ref struct`s have significant limitations:
    - They cannot be stored in arrays, fields of classes, or other heap-based collections
    - Implementing enumeration is more complex (C# 13 allows `ref struct` to implement `IEnumerable` but is complex). 

  Ultimately using pointers seems a better choice because:
    - ref structs are designed to support safe references to managed memory (possibly subject to relocation by GC)
      but in our special case we can take advantage of fact that the graph nodes are fixed in memory.
    - these feature limitations are quite unhelpful for code modularity and clarity 
    - the use of pointer is strictly encapsulated to this structure
    - any bugs resulting in attempts to dereference invalid/stale pointers 
      will be likely be detected, because upon graph deallocation the virtual memory address range is 
      unlikely to be used in subsequent allocations, especially with modern ASLR (Address Space Layout Randomization)
