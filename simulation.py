"""
=============================================================
  CSE112 - Simulation Framework
  Cairo Smart City Transportation Network Optimization
=============================================================

Provides a unified testing framework that runs all algorithms
under different scenarios and measures their performance.

Scenarios:
    1. Normal conditions      — afternoon, low traffic
    2. Morning rush hour      — morning peak, high congestion
    3. Evening rush hour      — evening peak, high congestion
    4. Road closure           — one road blocked, rerouting tested
    5. Emergency response     — ambulance routing to hospitals
    6. Night free flow        — night, minimal traffic

Usage:
    python simulation.py               # run all scenarios
    python simulation.py --scenario 3  # run one scenario
"""

import os
import sys
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_loader1  import CairoGraph
from dijkstra      import get_shortest_path, get_shortest_path_cached, clear_route_cache
from astar         import get_emergency_path, compare_algorithms


def p(f): return os.path.join(BASE_DIR, f)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def build_graph() -> CairoGraph:
    graph = CairoGraph()
    graph.load_data(p("nodes.csv"), p("edges.csv"))
    try:
        from traffic_ml_model import build_predictor
        csv = (p("Traffic_Flow_Patterns_live.csv")
               if os.path.exists(p("Traffic_Flow_Patterns_live.csv"))
               else p("Traffic_Flow_Patterns.csv"))
        graph.attach_predictor(build_predictor(csv, p("edges.csv"), p("nodes.csv")))
        print("✅ ML predictor attached.")
    except Exception as e:
        print(f"⚠️  ML not attached ({e}). Using distance fallback.")
    return graph


def separator(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")


# ─────────────────────────────────────────────
# PERFORMANCE MEASUREMENT
# ─────────────────────────────────────────────

def measure(fn, *args) -> tuple:
    """Run fn(*args) and return (result, elapsed_ms)."""
    t0     = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - t0) * 1000, 3)


# ─────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────

def scenario_normal(graph):
    separator("SCENARIO 1 — Normal Conditions (Afternoon)")
    pairs = [("1","2"),("3","7"),("4","F9"),("8","11"),("13","3")]
    _run_dijkstra_table(graph, pairs, "afternoon")


def scenario_morning_rush(graph):
    separator("SCENARIO 2 — Morning Rush Hour")
    pairs = [("1","2"),("3","7"),("4","F9"),("8","11"),("13","3")]
    _run_dijkstra_table(graph, pairs, "morning")


def scenario_evening_rush(graph):
    separator("SCENARIO 3 — Evening Rush Hour")
    pairs = [("1","2"),("3","7"),("4","F9"),("8","11"),("13","3")]
    _run_dijkstra_table(graph, pairs, "evening")


def scenario_road_closure(graph):
    separator("SCENARIO 4 — Road Closure (Downtown ↔ Nasr City blocked)")

    # Temporarily remove the 3↔2 edge
    original = graph.adjacency_list.get("3", []).copy()
    graph.adjacency_list["3"] = [e for e in original if e.to_id != "2"]
    graph.adjacency_list["2"] = [e for e in graph.adjacency_list.get("2",[]) if e.to_id != "3"]
    clear_route_cache()

    print("\n  Road 3 ↔ 2 CLOSED — alternate routes:")
    for src, tgt in [("4","3"),("1","2"),("5","3")]:
        r, ms = measure(get_shortest_path, graph, src, tgt, "morning")
        fn = graph.get_node(src).name
        tn = graph.get_node(tgt).name
        status = "✅ found" if r.found else "❌ no path"
        print(f"  {fn} → {tn}: {' → '.join(r.path_names(graph))} "
              f"| {r.total_time:.1f}min | {status} | {ms}ms")

    # Restore
    graph.adjacency_list["3"] = original
    clear_route_cache()
    print("\n  Road restored.")


def scenario_emergency(graph):
    separator("SCENARIO 5 — Emergency Response (Dijkstra vs A*)")

    test_pairs = [
        ("1",  "F9",  "Maadi → Qasr El Aini"),
        ("4",  "F9",  "New Cairo → Qasr El Aini"),
        ("7",  "F10", "6th October → Maadi Military"),
        ("13", "F9",  "New Admin Cap → Qasr El Aini"),
        ("12", "F10", "Helwan → Maadi Military"),
    ]

    print(f"\n  {'Route':<40} {'Dijk(ms)':>9} {'A*(ms)':>8} "
          f"{'D.nodes':>8} {'A*.nodes':>9} {'Saved':>6} {'Same?':>6}")
    print(f"  {'-'*40} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*6} {'-'*6}")

    for src, tgt, label in test_pairs:
        d_res, d_ms = measure(get_shortest_path,  graph, src, tgt, "morning")
        a_res, a_ms = measure(get_emergency_path, graph, src, tgt, "morning")
        saved = d_res.nodes_explored - a_res.nodes_explored
        same  = "✅" if d_res.path == a_res.path else "❌"
        print(f"  {label:<40} {d_ms:>9.2f} {a_ms:>8.2f} "
              f"{d_res.nodes_explored:>8} {a_res.nodes_explored:>9} {saved:>6} {same:>6}")


def scenario_night(graph):
    separator("SCENARIO 6 — Night Free Flow")
    pairs = [("1","2"),("3","7"),("4","F9"),("8","11"),("13","3")]
    _run_dijkstra_table(graph, pairs, "night")


def scenario_memoization(graph):
    separator("SCENARIO 7 — Memoization Performance Test")

    pairs = [("1","2"),("3","7"),("4","F9"),("8","F10"),("13","3")]
    print("\n  First call (cold cache) vs Second call (warm cache):\n")
    print(f"  {'Route':<35} {'Cold (ms)':>10} {'Warm (ms)':>10} {'Speedup':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    clear_route_cache()
    for src, tgt in pairs:
        _, cold = measure(get_shortest_path_cached, graph, src, tgt, "morning")
        _, warm = measure(get_shortest_path_cached, graph, src, tgt, "morning")
        speedup = round(cold / warm, 1) if warm > 0 else "∞"
        fn = graph.get_node(src).name
        tn = graph.get_node(tgt).name
        print(f"  {fn[:15]} → {tn[:15]:<15} {cold:>10.3f} {warm:>10.3f} {str(speedup):>9}x")

    clear_route_cache()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _run_dijkstra_table(graph, pairs, time_of_day):
    print(f"\n  {'From':<25} {'To':<25} {'Time (min)':>10}  {'Path'}")
    print(f"  {'-'*25} {'-'*25} {'-'*10}  {'-'*35}")
    for src, tgt in pairs:
        r, ms = measure(get_shortest_path, graph, src, tgt, time_of_day)
        fn    = graph.get_node(src).name
        tn    = graph.get_node(tgt).name
        names = " → ".join(r.path_names(graph)) if r.found else "No path"
        print(f"  {fn:<25} {tn:<25} {r.total_time:>10.1f}  {names}  ({ms}ms)")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

SCENARIOS = {
    1: ("Normal Conditions",      scenario_normal),
    2: ("Morning Rush Hour",      scenario_morning_rush),
    3: ("Evening Rush Hour",      scenario_evening_rush),
    4: ("Road Closure",           scenario_road_closure),
    5: ("Emergency Response",     scenario_emergency),
    6: ("Night Free Flow",        scenario_night),
    7: ("Memoization Test",       scenario_memoization),
}

def main():
    parser = argparse.ArgumentParser(description="Cairo Smart City — Simulation Framework")
    parser.add_argument("--scenario", type=int, choices=SCENARIOS.keys(),
                        help="Run a specific scenario only")
    args = parser.parse_args()

    print("\n🧪  Cairo Smart City — Simulation Framework")
    graph = build_graph()

    if args.scenario:
        label, fn = SCENARIOS[args.scenario]
        fn(graph)
    else:
        for num, (label, fn) in SCENARIOS.items():
            fn(graph)

    print("\n✅  Simulation complete.\n")


if __name__ == "__main__":
    main()
