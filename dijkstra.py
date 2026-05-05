"""
=============================================================
  CSE112 - Member 3: Dijkstra's Shortest Path Algorithm
  Cairo Smart City Transportation Network Optimization
=============================================================

Finds the fastest route between any two neighborhoods/facilities
using dynamic ML-powered weights from Member 1's engine.

Public API:
    dijkstra(graph, source_id, time_of_day) -> (distances, previous)
    get_shortest_path(graph, source_id, target_id, time_of_day) -> PathResult
"""

import heapq
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


# ─────────────────────────────────────────────
# 1. RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class PathResult:
    source_id:    str
    target_id:    str
    time_of_day:  str
    path:         list          # list of node IDs from source → target
    total_time:   float         # total travel time in minutes
    found:        bool          # False if no path exists
    nodes_explored: int = 0     # how many nodes Dijkstra visited (for animation)
    exploration_order: list = field(default_factory=list)  # for A* race animation

    def path_names(self, graph) -> list:
        """Return human-readable list of node names along the path."""
        return [graph.get_node(nid).name for nid in self.path if graph.get_node(nid)]

    def summary(self, graph):
        """Print a human-readable trip summary."""
        src  = graph.get_node(self.source_id)
        tgt  = graph.get_node(self.target_id)
        print("\n" + "═" * 55)
        print(f"  🗺️  TRIP PLANNER  —  {self.time_of_day.upper()}")
        print("═" * 55)
        if not self.found:
            print(f"  ❌  No path found from {src.name} to {tgt.name}")
            return
        print(f"  From : {src.name}")
        print(f"  To   : {tgt.name}")
        print(f"  Time : {self.time_of_day.capitalize()}")
        print(f"  ─────────────────────────────────────────────")
        names = self.path_names(graph)
        for i, name in enumerate(names):
            prefix = "  🚦 " if i == 0 else ("  🏁 " if i == len(names)-1 else "  ➜  ")
            print(f"{prefix}{name}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  ⏱️  Estimated travel time: {self.total_time:.1f} minutes")
        print(f"  🔍  Nodes explored      : {self.nodes_explored}")
        print("═" * 55)


# ─────────────────────────────────────────────
# 2. CORE DIJKSTRA ALGORITHM
# ─────────────────────────────────────────────

def dijkstra(graph, source_id: str, time_of_day: str = "morning"):
    """
    Run Dijkstra's algorithm from a source node across the entire graph.

    Args:
        graph       : CairoGraph instance (from ai_model.py)
        source_id   : Starting node ID (e.g. "3" or "F9")
        time_of_day : "morning" | "afternoon" | "evening" | "night"

    Returns:
        distances   : { node_id: float }  — shortest time (minutes) to every node
        previous    : { node_id: str|None } — previous node in optimal path
        order       : [ node_id, ... ]    — exploration order (for animation)
    """
    source_id = str(source_id)

    # Initialise all distances to infinity
    distances = {nid: float('inf') for nid in graph.nodes}
    distances[source_id] = 0.0

    previous = {nid: None for nid in graph.nodes}
    visited  = set()
    order    = []   # exploration order for animation

    # Priority queue: (cost, node_id)
    pq = [(0.0, source_id)]

    while pq:
        current_dist, current_id = heapq.heappop(pq)

        # Skip if already settled (lazy deletion)
        if current_id in visited:
            continue
        visited.add(current_id)
        order.append(current_id)

        # Explore neighbors via existing roads only
        for edge in graph.get_edges_from(current_id, existing_only=True):
            neighbor_id = edge.to_id

            if neighbor_id in visited:
                continue

            # Get ML-powered dynamic weight (travel time in minutes)
            weight = graph.get_weight(current_id, neighbor_id, time_of_day)

            new_dist = current_dist + weight
            if new_dist < distances.get(neighbor_id, float('inf')):
                distances[neighbor_id] = new_dist
                previous[neighbor_id]  = current_id
                heapq.heappush(pq, (new_dist, neighbor_id))

    return distances, previous, order


# ─────────────────────────────────────────────
# 3. PATH RECONSTRUCTION HELPER
# ─────────────────────────────────────────────

def _reconstruct_path(previous: dict, source_id: str, target_id: str) -> list:
    """Walk the 'previous' dict backwards to build the path."""
    path = []
    current = target_id

    while current is not None:
        path.append(current)
        current = previous.get(current)
        if current == source_id:
            path.append(source_id)
            break

    # If source not found → no path
    if path and path[-1] != source_id:
        return []

    path.reverse()
    return path


# ─────────────────────────────────────────────
# 4. MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def get_shortest_path(
    graph,
    source_id:   str,
    target_id:   str,
    time_of_day: str = "morning",
) -> PathResult:
    """
    Find the shortest (fastest) path between two nodes.

    Args:
        graph       : CairoGraph instance
        source_id   : Origin node ID
        target_id   : Destination node ID
        time_of_day : "morning" | "afternoon" | "evening" | "night"

    Returns:
        PathResult with .path, .total_time, .found, etc.
    """
    source_id = str(source_id)
    target_id = str(target_id)

    # Validate nodes exist
    if source_id not in graph.nodes or target_id not in graph.nodes:
        return PathResult(
            source_id=source_id, target_id=target_id,
            time_of_day=time_of_day, path=[], total_time=float('inf'), found=False
        )

    # Same source and destination
    if source_id == target_id:
        return PathResult(
            source_id=source_id, target_id=target_id,
            time_of_day=time_of_day, path=[source_id], total_time=0.0,
            found=True, nodes_explored=1, exploration_order=[source_id]
        )

    distances, previous, order = dijkstra(graph, source_id, time_of_day)

    total_time = distances.get(target_id, float('inf'))
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
# 5. UTILITY — ALL SHORTEST PATHS FROM SOURCE
# ─────────────────────────────────────────────

def get_all_paths_from(graph, source_id: str, time_of_day: str = "morning") -> dict:
    """
    Returns a dict of PathResult for every reachable node from source_id.
    Useful for the Trip Planner UI to show times to all destinations at once.
    """
    distances, previous, order = dijkstra(graph, source_id, time_of_day)
    results = {}

    for target_id in graph.nodes:
        if target_id == str(source_id):
            continue
        total_time = distances.get(target_id, float('inf'))
        found      = total_time < float('inf')
        path       = _reconstruct_path(previous, str(source_id), target_id) if found else []

        results[target_id] = PathResult(
            source_id=str(source_id),
            target_id=target_id,
            time_of_day=time_of_day,
            path=path,
            total_time=total_time,
            found=found,
            nodes_explored=len(order),
            exploration_order=order,
        )

    return results




# ─────────────────────────────────────────────
# MEMOIZED ROUTE CACHE
# ─────────────────────────────────────────────

_route_cache: dict = {}

def get_shortest_path_cached(graph, source_id: str, target_id: str,
                              time_of_day: str = "morning") -> "PathResult":
    """
    Memoized wrapper around get_shortest_path().
    Returns cached result if the same (source, target, time) was already computed.
    Cache is cleared automatically when the graph's ML predictor changes.
    """
    key = (str(source_id), str(target_id), time_of_day)
    if key not in _route_cache:
        _route_cache[key] = get_shortest_path(graph, source_id, target_id, time_of_day)
    return _route_cache[key]

def clear_route_cache():
    """Call this if the graph weights change (e.g. after attaching a new predictor)."""
    _route_cache.clear()

# ─────────────────────────────────────────────
# 6. TESTING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    # Add the folder containing ai_model.py to path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE_DIR)

    from data_loader1 import CairoGraph

    # ── Setup ────────────────────────────────────────────
    def p(f): return os.path.join(BASE_DIR, f)

    graph = CairoGraph()
    graph.load_data(p("nodes.csv"), p("edges.csv"))

    # Attach ML predictor (Member 1's model)
    try:
        from traffic_ml_model import build_predictor
        predictor = build_predictor(p("Traffic_Flow_Patterns.csv"), p("edges.csv"), p("nodes.csv"))
        graph.attach_predictor(predictor)
    except Exception as e:
        print(f"⚠️  ML predictor not attached ({e}). Using distance fallback.")

    graph.summary()

    # ── Test 1: Single route ─────────────────────────────
    print("\n" + "="*55)
    print("  TEST 1: Maadi → Nasr City (Morning)")
    result = get_shortest_path(graph, "1", "2", "morning")
    result.summary(graph)

    # ── Test 2: Emergency route to hospital ─────────────
    print("\n  TEST 2: New Cairo → Qasr El Aini Hospital (Morning)")
    result = get_shortest_path(graph, "4", "F9", "morning")
    result.summary(graph)

    # ── Test 3: Same route, night vs morning ─────────────
    print("\n  TEST 3: Downtown Cairo → 6th October City")
    for slot in ["morning", "night"]:
        r = get_shortest_path(graph, "3", "7", slot)
        names = r.path_names(graph)
        print(f"  [{slot.upper():>9}]  {' → '.join(names)}")
        print(f"             ⏱  {r.total_time:.1f} min  |  {r.nodes_explored} nodes explored")

    # ── Test 4: All destinations from Downtown ───────────
    print("\n  TEST 4: All paths from Downtown Cairo (Evening)")
    all_paths = get_all_paths_from(graph, "3", "evening")
    sorted_paths = sorted(all_paths.values(), key=lambda r: r.total_time)
    print(f"  {'Destination':<35} {'Time (min)':>10}  Path")
    print(f"  {'-'*35} {'-'*10}  {'-'*30}")
    for r in sorted_paths:
        if r.found:
            node = graph.get_node(r.target_id)
            names = r.path_names(graph)
            print(f"  {node.name:<35} {r.total_time:>10.1f}  {' → '.join(names)}")
