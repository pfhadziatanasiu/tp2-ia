import heapq
from typing import Optional, List, Tuple, Dict, Set

# -----------------------------
# Node definition
# -----------------------------
class Node:
    def __init__(self, state: int, parent: Optional["Node"] = None,
                 action: Optional[int] = None, depth: int = 0, cost: int = 0):
        """
        Search-tree node.
        - state: current integer position on the horizontal line H.
        - parent: pointer to the parent Node (None for the root).
        - action: action APPLIED TO THE PARENT that produced THIS node.
                  Here: +1 (move right) or -1 (move left). None for the root.
        - depth: depth in the tree (root = 0).
        - cost:  path cost g(n). Here we use unit step cost => number of moves.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.cost = cost  # g(n)

    def __repr__(self) -> str:
        return (f"Node(state={self.state}, action={self.action}, "
                f"depth={self.depth}, cost={self.cost})")


# -----------------------------
# Successor function (children)
# -----------------------------
def successors_1d(x: int) -> List[Tuple[int, int]]:
    """
    From state x, you can go to:
      - (x - 1) via action -1
      - (x + 1) via action +1
    Returns a list of (next_state, action).
    """
    return [(x - 1, -1), (x + 1, +1)]


# -----------------------------
# Heuristic h(n) = |goal - x|
# -----------------------------
def h_abs_distance(x: int, goal: int) -> int:
    """Admissible and consistent in 1D with unit steps."""
    return abs(goal - x)


# -----------------------------
# Path reconstruction using parent pointers
# -----------------------------
def reconstruct_path(goal_node: Node) -> List[Tuple[int, Optional[int]]]:
    """
    Follow parent pointers from goal to root.
    Return a list of (state, action_that_produced_this_state),
    ordered from start to goal. For the root, action is None.
    """
    path: List[Tuple[int, Optional[int]]] = []
    n: Optional[Node] = goal_node
    while n is not None:
        path.append((n.state, n.action))
        n = n.parent
    path.reverse()  # now it goes start -> ... -> goal
    return path


# -----------------------------
# A* for the 1D ±1 problem
# -----------------------------
def astar_1d(start: int, goal: int, min_state: int, max_state: int) -> Optional[List[Tuple[int, Optional[int]]]]:
    """
    A* on a 1D integer line using heapq:
      - OPEN: min-heap ordered by f = g + h
      - OPEN_BEST_F: state -> best f known while in OPEN (to skip stale heap entries)
      - CLOSED_F: state -> best f already expanded
      - Reopen if a strictly better f appears for a state in CLOSED
    """
    print(f"[INIT ] start={start}, goal={goal}, bounds=[{min_state}, {max_state}]")

    root = Node(state=start, parent=None, action=None, depth=0, cost=0)  # g=0
    f_root = root.cost + h_abs_distance(root.state, goal)

    # OPEN = min-heap of (f, tie_breaker, node)
    open_heap: List[Tuple[int, int, Node]] = []
    heapq.heappush(open_heap, (f_root, root.depth, root))

    # Track best f in OPEN (to detect/skip stale entries from the heap)
    open_best_f: Dict[int, int] = {start: f_root}

    # CLOSED stores best f already expanded for each state
    closed_f: Dict[int, int] = {}

    print(f"[OPEN ] push root: state={start}, g={root.cost}, h={h_abs_distance(start, goal)}, f={f_root}")

    while open_heap:
        f_curr, _, current = heapq.heappop(open_heap)

        # Skip heap entries that are stale (a better f for this state is in OPEN)
        if open_best_f.get(current.state, float("inf")) < f_curr:
            print(f"[SKIP ] stale entry for state={current.state} (better f already in OPEN)")
            continue

        print(f"\n[POP  ] state={current.state}, g={current.cost}, "
              f"h={h_abs_distance(current.state, goal)}, f={f_curr}, depth={current.depth}")

        # Goal test
        if current.state == goal:
            print("[GOAL ] goal reached. Reconstructing path via parent pointers...")
            return reconstruct_path(current)

        # Move to CLOSED with its best f
        closed_f[current.state] = f_curr
        print(f"[CLOSE] add state={current.state} with f={f_curr}")
        print(f"        CLOSED now: {closed_f}")

        # Expand successors
        for nxt_state, act in successors_1d(current.state):
            # Bounds
            if nxt_state < min_state or nxt_state > max_state:
                print(f"[PRUNE] out-of-bounds successor {nxt_state} via action {act}")
                continue

            g_child = current.cost + 1                      # unit step cost
            h_child = h_abs_distance(nxt_state, goal)
            f_child = g_child + h_child

            # Compare against CLOSED by f (course requirement)
            prev_f_closed = closed_f.get(nxt_state)
            if prev_f_closed is not None:
                if f_child >= prev_f_closed:
                    print(f"[SEEN ] in CLOSED with better/equal f: state={nxt_state} "
                          f"(f_new={f_child} >= f_closed={prev_f_closed})")
                    continue
                else:
                    # Reopen
                    print(f"[REOPEN] state={nxt_state}: f_new={f_child} < f_closed={prev_f_closed} -> reopen")
                    del closed_f[nxt_state]

            # Compare against OPEN by f (keep only best f per state)
            prev_f_open = open_best_f.get(nxt_state)
            if prev_f_open is not None and f_child >= prev_f_open:
                print(f"[SEEN ] in OPEN with better/equal f: state={nxt_state} "
                      f"(f_new={f_child} >= f_open={prev_f_open})")
                continue

            # Push improved child
            child = Node(state=nxt_state, parent=current, action=act,
                         depth=current.depth + 1, cost=g_child)  # store g for path/cost reporting
            heapq.heappush(open_heap, (f_child, child.depth, child))
            open_best_f[nxt_state] = f_child

            print(f"[PUSH ] child state={child.state} via action={act} "
                  f"g={g_child}, h={h_child}, f={f_child}, depth={child.depth})")

    print("\n[FAIL ] no solution within given bounds")
    return None

# -----------------------------
# Example run (you can change values)
# -----------------------------
if __name__ == "__main__":
    # Example: start at 0, goal at 2, restrict search to [-2, 2]
    path = astar_1d(start=0, goal=2, min_state=-2, max_state=2)
    if path is not None:
        print("\n[PATH] start -> goal (state, action_that_produced_state):")
        for st, act in path:
            print(f"  state={st}, action={act}")
    else:
        print("[INFO] No path found.")