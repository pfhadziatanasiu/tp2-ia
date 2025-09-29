import heapq
from typing import Optional, List, Tuple, Dict

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
        - action: operator applied to the parent to produce this node.
        - depth: depth in the search tree (root = 0).
        - cost: path cost g(n). With unit step costs, cost == depth.
                With proportional action costs (|a|), cost may differ from depth.
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
# Successor function
# -----------------------------
def successors_1d(x: int, actions: List[int]) -> List[Tuple[int, int]]:
    """
    Apply each operator in actions to state x.
    Returns [(next_state, action), ...].
    """
    return [(x + a, a) for a in actions]


# -----------------------------
# Heuristic h(n) = |goal - x|
# -----------------------------
def h_abs_distance(x: int, goal: int) -> int:
    """Admissible and consistent when action cost is proportional to |a|."""
    return abs(goal - x)


# -----------------------------
# Path reconstruction
# -----------------------------
def reconstruct_path(goal_node: Node) -> List[Tuple[int, Optional[int]]]:
    path: List[Tuple[int, Optional[int]]] = []
    n: Optional[Node] = goal_node
    while n is not None:
        path.append((n.state, n.action))
        n = n.parent
    path.reverse()
    return path


def print_open(open_heap, open_best_f):
    """
    Pretty-print OPEN as a sorted snapshot of the heap entries (state, f, depth).
    Excludes stale entries (worse than the best known f for that state).
    Note: Debug view only; does not modify the heap.
    """
    snapshot = []
    for f, depth, state, _node in open_heap:
        # keep only non-stale entries
        if f <= open_best_f.get(state, float("inf")):
            snapshot.append((state, f, depth))
    snapshot.sort(key=lambda t: (t[1], t[2], t[0]))  # by f, then depth, then state
    pretty = [f"(state={s}, f={f}, depth={d})" for (s, f, d) in snapshot]
    print("[OPEN] now: ", "[" + ", ".join(pretty) + "]")


# -----------------------------
# A* with configurable actions (OPEN/CLOSED by f = g + h)
# -----------------------------
def astar_1d(start: int, goal: int, min_state: int, max_state: int,
             actions: List[int]) -> Optional[List[Tuple[int, Optional[int]]]]:
    
    # --- Parameter validation ---
    for name, value in [("start", start), ("goal", goal),
                        ("min_state", min_state), ("max_state", max_state)]:
        if not isinstance(value, int):
            raise TypeError(f"Parameter '{name}' must be an integer, got {type(value).__name__}: {value}")

    if not isinstance(actions, list) or not all(isinstance(a, int) for a in actions):
        raise TypeError("Parameter 'actions' must be a list of integers")

    print(f"[INIT ] start={start}, goal={goal}, bounds=[{min_state}, {max_state}], actions={actions}")

    # Root node
    root = Node(state=start, parent=None, action=None, depth=0, cost=0)
    f_root = root.cost + h_abs_distance(root.state, goal)

    # OPEN = min-heap of (f, tie_breaker_depth, tie_breaker_state, node)
    open_heap: List[Tuple[int, int, int, Node]] = []
    heapq.heappush(open_heap, (f_root, root.depth, root.state, root))

    # Track best f in OPEN (to detect/skip stale heap entries)
    open_best_f: Dict[int, int] = {start: f_root}

    # CLOSED: per course requirement we track best f that has been expanded per state
    closed_f: Dict[int, int] = {}

    print(f"[OPEN ] insert root: state={start}, g={root.cost}, h={h_abs_distance(start, goal)}, f={f_root}")

    while open_heap:
        # Show OPEN at the start of the iteration (before extract)
        print_open(open_heap, open_best_f)

        # Extract node with smallest f
        f_curr, _, _, current = heapq.heappop(open_heap)
        # Keep OPEN map consistent with the heap view
        open_best_f.pop(current.state, None)

        # Skip stale heap entries (a better f for this state is already in OPEN)
        # (If we removed its map entry above, a stale one won't pass this check anyway)
        if open_best_f.get(current.state, float("inf")) < f_curr:
            print(f"[SKIP ] stale entry for state={current.state} (better f already in OPEN)")
            continue

        print(f"\n[EXTRACT] state={current.state}, g={current.cost}, "
              f"h={h_abs_distance(current.state, goal)}, f={f_curr}, depth={current.depth}")

        # Goal test (goal-on-extract)
        if current.state == goal:
            print("[GOAL ] goal reached. Reconstructing path...")
            return reconstruct_path(current)

        # Move to CLOSED with its best f (course requirement)
        closed_f[current.state] = f_curr
        print(f"[CLOSE] add state={current.state} with f={f_curr}")
        print(f"[CLOSED] {closed_f}")

        # Expand successors
        for nxt_state, act in successors_1d(current.state, actions):
            # Bounds
            if nxt_state < min_state or nxt_state > max_state:
                print(f"[PRUNE] out-of-bounds successor {nxt_state} via action {act}")
                continue

            # Path cost update: proportional to step length
            g_child = current.cost + abs(act)
            h_child = h_abs_distance(nxt_state, goal)
            f_child = g_child + h_child

            # Compare against CLOSED by f (course requirement)
            prev_f_closed = closed_f.get(nxt_state)
            if prev_f_closed is not None and f_child >= prev_f_closed:
                print(f"[SEEN ] in CLOSED with better/equal f: state={nxt_state} "
                      f"(f_new={f_child} >= f_closed={prev_f_closed})")
                continue
            elif prev_f_closed is not None and f_child < prev_f_closed:
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
                         depth=current.depth + 1, cost=g_child)
            heapq.heappush(open_heap, (f_child, child.depth, child.state, child))
            open_best_f[nxt_state] = f_child

            print(f"[INSERT] child state={child.state} via action={act} "
                  f"g={g_child}, h={h_child}, f={f_child}, depth={child.depth}")
            print_open(open_heap, open_best_f)

    print("\n[FAIL ] no solution within given bounds")
    return None


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    # Pass actions explicitly (e.g., allow +/-1 and +/-2 steps)
    path = astar_1d(start=0, goal=2, min_state=-2, max_state=2, actions=[-2, -1, 1, 2])
    if path is not None:
        print("\n[PATH] start -> goal (state, action_that_produced_state):")
        for st, act in path:
            print(f"  state={st}, action={act}")
    else:
        print("[INFO] No path found.")