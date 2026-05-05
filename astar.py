"""
=============================================================
  CSE112 - Member 4: A* Search Algorithm
  Cairo Smart City Transportation Network Optimization
=============================================================

Finds the fastest emergency route between any two nodes using
A* search with a Euclidean-distance heuristic on X/Y coords.
Designed for routing emergency vehicles to medical facilities
(F9 – Qasr El Aini Hospital, F10 – Maadi Military Hospital).

Compared to Dijkstra, A* is guided by h(n), the straight-line
distance to the goal, so it explores fewer nodes — proven by
the side-by-side race animation in the UI.

Public API:
    astar(graph, source_id, target_id, time_of_day)
        -> (g_costs, previous, order)

    get_emergency_path(graph, source_id, target_id, time_of_day)
        -> PathResult          (same dataclass as Dijkstra)

    compare_algorithms(graph, source_id, target_id, time_of_day)
        -> dict with both PathResults for the race animation
"""

import heapq
import math

# Re-use Dijkstra's PathResult so the UI can treat both uniformly
from dijkstra import PathResult, _reconstruct_path


# ─────────────────────────────────────────────
# 1. HEURISTIC
# ─────────────────────────────────────────────

def euclidean_heuristic(graph, node_id: str, goal_id: str) -> float:
    """
    Admissible heuristic h(n): straight-line (Euclidean) distance
    between node_id and goal_id, converted to an optimistic travel-
    time estimate (assumes free-flow speed of ~1 min/km).

    Because real travel time >= distance, h(n) never over-estimates,
    guaranteeing A* finds the optimal path.

    Coordinate system: nodes store longitude in .x, latitude in .y
        1 degree latitude  ≈ 111.0 km
        1 degree longitude ≈  96.3 km  (at Cairo's latitude ~30°)
    """
    node = graph.get_node(node_id)
    goal = graph.get_node(goal_id)
    if node is None or goal is None:
        return 0.0

    dlat = (goal.y - node.y) * 111.0
    dlon = (goal.x - node.x) * 96.3
    dist_km = math.sqrt(dlat ** 2 + dlon ** 2)

    # Optimistic: 1 min per km (free-flow, zero congestion)
    return dist_km


# ─────────────────────────────────────────────
# 2. CORE A* ALGORITHM
# ─────────────────────────────────────────────

def astar(graph, source_id: str, target_id: str, time_of_day: str = "morning"):
    """
    Run A* search from source_id to target_id.

    Priority queue key: f(n) = g(n) + h(n)
        g(n) — actual cost from source (ML travel time in minutes)
        h(n) — Euclidean lower-bound to goal (admissible heuristic)

    A* expands the node most likely to be on the cheapest path,
    stopping as soon as the goal is popped from the queue.

    Args:
        graph       : CairoGraph instance (from data_loader1.py)
        source_id   : Starting node ID
        target_id   : Goal node ID
        time_of_day : "morning" | "afternoon" | "evening" | "night"

    Returns:
        g_cost   : { node_id: float } — best g-cost found per node
        previous : { node_id: str|None } — predecessor in optimal path
        order    : [ node_id, ... ]   — expansion order (for animation)
    """
    source_id = str(source_id)
    target_id = str(target_id)

    g_cost   = {nid: float('inf') for nid in graph.nodes}
    g_cost[source_id] = 0.0

    previous = {nid: None for nid in graph.nodes}
    closed   = set()
    order    = []

    # Tie-breaker counter prevents comparing node_id strings in heapq
    counter = 0
    h0      = euclidean_heuristic(graph, source_id, target_id)
    pq      = [(h0, counter, source_id)]   # (f, tie, node_id)

    while pq:
        f, _, current_id = heapq.heappop(pq)

        if current_id in closed:
            continue
        closed.add(current_id)
        order.append(current_id)

        # Early exit: optimal path to goal is confirmed
        if current_id == target_id:
            break

        for edge in graph.get_edges_from(current_id, existing_only=True):
            neighbor_id = edge.to_id
            if neighbor_id in closed:
                continue

            weight = graph.get_weight(current_id, neighbor_id, time_of_day)
            new_g  = g_cost[current_id] + weight

            if new_g < g_cost.get(neighbor_id, float('inf')):
                g_cost[neighbor_id]   = new_g
                previous[neighbor_id] = current_id
                h       = euclidean_heuristic(graph, neighbor_id, target_id)
                counter += 1
                heapq.heappush(pq, (new_g + h, counter, neighbor_id))

    return g_cost, previous, order


# ─────────────────────────────────────────────
# 3. MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def get_emergency_path(
    graph,
    source_id:   str,
    target_id:   str,
    time_of_day: str = "morning",
) -> PathResult:
    """
    Find the fastest emergency route between two nodes using A*.

    Identical interface to dijkstra.get_shortest_path() so the UI
    can call both interchangeably.

    Args:
        graph       : CairoGraph instance
        source_id   : Origin node ID
        target_id   : Destination node ID (typically F9 or F10)
        time_of_day : "morning" | "afternoon" | "evening" | "night"

    Returns:
        PathResult with .path, .total_time, .found, .nodes_explored
    """
    source_id = str(source_id)
    target_id = str(target_id)

    if source_id not in graph.nodes or target_id not in graph.nodes:
        return PathResult(
            source_id=source_id, target_id=target_id,
            time_of_day=time_of_day, path=[], total_time=float('inf'), found=False
        )

    if source_id == target_id:
        return PathResult(
            source_id=source_id, target_id=target_id,
            time_of_day=time_of_day, path=[source_id], total_time=0.0,
            found=True, nodes_explored=1, exploration_order=[source_id]
        )

    g_cost, previous, order = astar(graph, source_id, target_id, time_of_day)

    total_time = g_cost.get(target_id, float('inf'))
    found      = total_time < float('inf')
    path       = _reconstruct_path(previous, source_id, target_id) if found else []

    return PathResult(
        source_id=source_id,
        target_id=target_id,
        time_of_day=time_of_day,
        path=path,
        total_time=total_time,
        found=found,
        nodes_explored=len(order),
        exploration_order=order,
    )


# ─────────────────────────────────────────────
# 4. ALGORITHM COMPARATOR  (for race animation)
# ─────────────────────────────────────────────

def compare_algorithms(
    graph,
    source_id:   str,
    target_id:   str,
    time_of_day: str = "morning",
) -> dict:
    """
    Run both Dijkstra and A* on the same query and return a
    side-by-side comparison dict.  Used by Member 4's race animation UI.

    Returns:
        {
            'dijkstra':    PathResult,
            'astar':       PathResult,
            'nodes_saved': int,   — how many fewer nodes A* explored
            'time_diff':   float, — path-time difference (should be ~0)
            'same_path':   bool,  — True if both found identical routes
        }
    """
    from dijkstra import get_shortest_path

    d_result = get_shortest_path(graph, source_id, target_id, time_of_day)
    a_result = get_emergency_path(graph, source_id, target_id, time_of_day)

    return {
        'dijkstra':    d_result,
        'astar':       a_result,
        'nodes_saved': d_result.nodes_explored - a_result.nodes_explored,
        'time_diff':   round(abs(d_result.total_time - a_result.total_time), 4),
        'same_path':   d_result.path == a_result.path,
    }


# ─────────────────────────────────────────────
# 5. TESTING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE_DIR)

    from data_loader1 import CairoGraph

    def p(f): return os.path.join(BASE_DIR, f)

    graph = CairoGraph()
    graph.load_data(p("nodes.csv"), p("edges.csv"))

    try:
        from traffic_ml_model import build_predictor
        predictor = build_predictor(
            p("Traffic_Flow_Patterns.csv"), p("edges.csv"), p("nodes.csv")
        )
        graph.attach_predictor(predictor)
    except Exception as e:
        print(f"⚠️  ML predictor not attached ({e}). Using distance fallback.")

    # ── Test 1: Emergency route to Qasr El Aini ──────────────────
    print("\n" + "="*55)
    print("  TEST 1: A* — New Cairo → Qasr El Aini Hospital (F9)")
    result = get_emergency_path(graph, "4", "F9", "morning")
    result.summary(graph)

    # ── Test 2: Algorithm race ────────────────────────────────────
    print("\n  TEST 2: Dijkstra vs A* Race — Giza → F9 (Morning)")
    cmp = compare_algorithms(graph, "8", "F9", "morning")
    d, a = cmp['dijkstra'], cmp['astar']
    print(f"\n  {'Algorithm':<12} {'Nodes Explored':>15} {'Travel Time':>12}  Path")
    print(f"  {'-'*12} {'-'*15} {'-'*12}  {'-'*30}")
    for label, r in [("Dijkstra", d), ("A*", a)]:
        names = r.path_names(graph)
        print(f"  {label:<12} {r.nodes_explored:>15}  {r.total_time:>10.1f}m  "
              f"{' → '.join(names)}")
    print(f"\n  🏆 A* explored {cmp['nodes_saved']} fewer nodes")
    print(f"  ✅ Same optimal path: {cmp['same_path']}")

    # ── Test 3: All time slots ────────────────────────────────────
    print("\n  TEST 3: Downtown Cairo → Maadi Military Hospital (F10)")
    for slot in ["morning", "afternoon", "evening", "night"]:
        r = get_emergency_path(graph, "3", "F10", slot)
        print(f"  [{slot.upper():>9}]  {r.total_time:.1f} min  |  "
              f"{r.nodes_explored} nodes  |  {' → '.join(r.path_names(graph))}")