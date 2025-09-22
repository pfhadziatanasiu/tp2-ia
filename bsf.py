from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set

@dataclass
class Node:
    """Search tree node as in AIMA-style: state, parent, action, depth, path cost."""
    state: int                     # current position on the horizontal line H
    parent: Optional["Node"]       # pointer to parent node (None for root)
    action: Optional[int]          # action applied to parent to obtain this node: +1 or -1 (None for root)
    depth: int                     # depth in the tree (root = 0)
    cost: int                      # path cost g(n); here we use 1 per step

def reconstruct_path(goal_node: Node) -> List[Tuple[int, Optional[int]]]:
    """
    Follow parent pointers from goal to root and return the path as a list
    of (state, action_that_produced_state), from start to goal.
    For the root, action is None.
    """
    path: List[Tuple[int, Optional[int]]] = []
    n: Optional[Node] = goal_node
    while n is not None:
        path.append((n.state, n.action))
        n = n.parent
    path.reverse()
    return path

def successors_1d(x: int) -> List[Tuple[int, int]]:
    """
    Successor function for the 1D problem.
    From state x, you can go to x-1 via action -1, and to x+1 via action +1.
    Returns a list of (next_state, action).
    """
    return [(x - 1, -1), (x + 1, 1)]

def bfs_1d(start: int, goal: int, min_state: int, max_state: int) -> Optional[List[Tuple[int, Optional[int]]]]:
    """
    Breadth-First Search on a 1D integer line with actions ±1.
    - Uses a FIFO queue (deque) as the frontier.
    - Stores ONLY nodes (with parent pointers). No path lists in the queue.
    - Prints every expansion and each generated child.
    Bounds [min_state, max_state] avoid infinite expansion on the integers.
    """
    # 1) Frontier as FIFO queue with the root node
    root = Node(state=start, parent=None, action=None, depth=0, cost=0)
    frontier: deque[Node] = deque([root])

    # 2) Explored set of states to prevent revisiting
    explored: Set[int] = set()

    print(f"[INIT] start={start}, goal={goal}, bounds=[{min_state}, {max_state}]")
    print("[INIT] frontier <- [start]")

    # 3) Main loop
    while frontier:
        current = frontier.popleft()  # FIFO: take the oldest node
        print(f"\n[POP ] expand state={current.state} depth={current.depth} cost={current.cost}")

        # Goal test
        if current.state == goal:
            print("[GOAL] goal reached. Reconstructing path via parent pointers...")
            return reconstruct_path(current)

        # Mark as explored (we only add when we pop/expand)
        if current.state in explored:
            print(f"[SKIP] state {current.state} already explored")
            continue
        explored.add(current.state)

        # Expand successors from the CURRENT state
        for next_state, action in successors_1d(current.state):
            if next_state < min_state or next_state > max_state:
                print(f"[PRUNE] skip out-of-bounds successor {next_state} via action {action}")
                continue
            if next_state in explored:
                print(f"[SEEN] successor {next_state} via action {action} already explored")
                continue

            child = Node(
                state=next_state,
                parent=current,
                action=action,           # action APPLIED TO PARENT to obtain THIS node
                depth=current.depth + 1,
                cost=current.cost + 1,   # unit step cost
            )
            frontier.append(child)
            print(f"[PUSH] generated child state={child.state} via action={action} depth={child.depth} cost={child.cost}")

    print("\n[FAIL] no solution within given bounds")
    return None

# --- Example run ---
if __name__ == "__main__":
    path = bfs_1d(start=0, goal=2, min_state=-2, max_state=2)
    if path is not None:
        print("\n[PATH] start -> goal (state, action_that_produced_state):")
        for state, action in path:
            print(f"  state={state}, action={action}")
    else:
        print("[INFO] No path found.")