"""
=============================================================
  CSE112 — Cairo Smart City Transportation Network
  main.py  |  Integration & Demo Runner
=============================================================

Wires together all seven member modules:
    Member 1 — data_loader1.py              (CairoGraph)
    Member 1 — traffic_ml_model.py          (ML traffic predictor)
    Member 1 — traffic_api.py               (TomTom live weights)
    Member 2 — kruskal_mst.py               (Kruskal's MST)
    Member 3 — dijkstra.py                  (trip planner)
    Member 4 — astar.py                     (emergency routing + race)
    Member 5 — public_transit_scheduler.py  (DP bus allocation)
    Member 6 — dp_maintenance.py            (DP knapsack maintenance)
    Member 7 — greedy.py                    (traffic signal optimizer)

Run:
    python main.py                  # full demo
    python main.py --module 2       # only Member 2 (MST)
    python main.py --module 3       # only Member 3 (Dijkstra)
    python main.py --module 4       # only Member 4 (A*)
    python main.py --module 5       # only Member 5 (Transit DP)
    python main.py --module 6       # only Member 6 (Maintenance DP)
    python main.py --module 7       # only Member 7 (Greedy Signals)
    python main.py --live           # use TomTom live weights (needs API key)
    python main.py --quiet          # suppress verbose output
    python main.py --no-png         # skip saving PNG charts
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_loader1              import CairoGraph
from dijkstra                  import get_shortest_path, get_all_paths_from
from astar                     import get_emergency_path, compare_algorithms
from kruskal_mst               import plan_new_infrastructure, visualize_infrastructure
from public_transit_scheduler  import optimize_transit, plot_transit
from dp_maintenance            import load_edges, load_nodes, solve_maintenance, plot_condition_heatmap
from greedy                    import SignalController, plot_signal_dashboard


def p(f): return os.path.join(BASE_DIR, f)


# ═════════════════════════════════════════════════════════════════
# GRAPH SETUP
# ═════════════════════════════════════════════════════════════════

def build_graph(quiet: bool = False, use_live: bool = False) -> CairoGraph:
    """
    Load graph and attach the best available predictor:
        1. TomTom live weights  (if --live and API key set)
        2. ML model on static CSV  (default)
        3. Distance fallback  (if CSV missing)
    """
    graph = CairoGraph()
    graph.load_data(p("nodes.csv"), p("edges.csv"))

    # ── Option 1: TomTom live weights ────────────────────────────
    if use_live:
        try:
            from traffic_api import TomTomFetcher, attach_live_weights
            fetcher    = TomTomFetcher()
            cache_file = p("live_weights.json")

            if not fetcher.load_cache(cache_file):
                if not quiet:
                    print("🌐 Fetching live weights from TomTom...")
                fetcher.fetch_all_edges(graph)
                fetcher.save_cache(cache_file)
                fetcher.save_as_traffic_csv(p("Traffic_Flow_Patterns_live.csv"))

            attach_live_weights(graph, fetcher, time_of_day="morning")

            # Also train ML on live data for time-slot predictions
            live_csv = p("Traffic_Flow_Patterns_live.csv")
            if os.path.exists(live_csv):
                from traffic_ml_model import build_predictor
                predictor = build_predictor(live_csv, p("edges.csv"), p("nodes.csv"))
                graph.attach_predictor(predictor)
                if not quiet:
                    print("✅ Live TomTom weights + ML predictor active.")
            return graph

        except Exception as e:
            if not quiet:
                print(f"⚠️  TomTom live fetch failed ({e}). Falling back to ML model.")

    # ── Option 2: ML model on static CSV ─────────────────────────
    try:
        from traffic_ml_model import build_predictor

        # Prefer live CSV if it already exists from a previous run
        traffic_csv = (p("Traffic_Flow_Patterns_live.csv")
                       if os.path.exists(p("Traffic_Flow_Patterns_live.csv"))
                       else p("Traffic_Flow_Patterns.csv"))

        predictor = build_predictor(traffic_csv, p("edges.csv"), p("nodes.csv"))
        graph.attach_predictor(predictor)
        if not quiet:
            src = "live" if "live" in traffic_csv else "static"
            print(f"✅ ML predictor attached ({src} traffic data).")

    except FileNotFoundError:
        if not quiet:
            print("⚠️  Traffic CSV not found. Using distance-based weights.")
    except Exception as e:
        if not quiet:
            print(f"⚠️  ML predictor skipped: {e}")

    return graph


def separator(title: str = ""):
    line = "═" * 62
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(f"\n{line}")


# ═════════════════════════════════════════════════════════════════
# MEMBER 2 — MST
# ═════════════════════════════════════════════════════════════════

def demo_mst(graph: CairoGraph, save_png: bool = True):
    separator("MEMBER 2 — Kruskal's MST  |  Infrastructure Expansion Plan")

    result = plan_new_infrastructure(graph)
    chosen = result['chosen_roads']

    print(f"\n  New roads selected : {len(chosen)}")
    print(f"  Total cost         : {result['total_cost']:,.1f} M EGP")
    print(f"  Priority savings   : {result['savings']:,.1f} M EGP\n")
    print(f"  {'From':<30} {'To':<30} {'Cost (M EGP)':>12}")
    print(f"  {'-'*30} {'-'*30} {'-'*12}")
    for edge in chosen:
        frm = graph.get_node(edge.from_id).name
        to  = graph.get_node(edge.to_id).name
        print(f"  {frm:<30} {to:<30} {edge.cost:>12,.1f}")

    if save_png:
        fig = visualize_infrastructure(graph, chosen)
        out = p("mst_infrastructure.html")
        fig.write_html(out)
        print(f"\n  📊 Map saved → {out}")

    return result


# ═════════════════════════════════════════════════════════════════
# MEMBER 3 — DIJKSTRA
# ═════════════════════════════════════════════════════════════════

def demo_dijkstra(graph: CairoGraph):
    separator("MEMBER 3 — Dijkstra's Algorithm  |  Trip Planner")

    print("\n  [1] Maadi → Nasr City  (Morning)")
    get_shortest_path(graph, "1", "2", "morning").summary(graph)

    print("\n  [2] New Cairo → Qasr El Aini Hospital  (Morning)")
    get_shortest_path(graph, "4", "F9", "morning").summary(graph)

    separator("Morning vs Night — Downtown Cairo → 6th October City")
    for slot in ["morning", "night"]:
        r = get_shortest_path(graph, "3", "7", slot)
        print(f"  [{slot.upper():>9}]  {r.total_time:>6.1f} min  |  "
              f"{' → '.join(r.path_names(graph))}")

    separator("All destinations from Downtown Cairo (Evening)")
    all_paths = get_all_paths_from(graph, "3", "evening")
    reachable = sorted([r for r in all_paths.values() if r.found],
                       key=lambda r: r.total_time)
    print(f"\n  {'Destination':<35} {'Time (min)':>10}  Path")
    print(f"  {'-'*35} {'-'*10}  {'-'*35}")
    for r in reachable:
        node  = graph.get_node(r.target_id)
        names = " → ".join(r.path_names(graph))
        print(f"  {node.name:<35} {r.total_time:>10.1f}  {names}")


# ═════════════════════════════════════════════════════════════════
# MEMBER 4 — A*
# ═════════════════════════════════════════════════════════════════

def demo_astar(graph: CairoGraph):
    separator("MEMBER 4 — A* Search  |  Emergency Response Routing")

    for source, target, label in [
        ("4",  "F9",  "New Cairo → Qasr El Aini Hospital"),
        ("7",  "F10", "6th October City → Maadi Military Hospital"),
        ("13", "F9",  "New Admin. Capital → Qasr El Aini Hospital"),
    ]:
        print(f"\n  🚑 {label}  (Morning)")
        get_emergency_path(graph, source, target, "morning").summary(graph)

    separator("Dijkstra vs A* Race  |  Node Exploration Comparison")
    test_pairs = [
        ("8",  "F9",  "Giza → Qasr El Aini Hospital"),
        ("4",  "F9",  "New Cairo → Qasr El Aini Hospital"),
        ("7",  "F10", "6th October → Maadi Military Hospital"),
        ("13", "F9",  "New Admin. Capital → F9"),
    ]
    print(f"\n  {'Route':<42} {'Dijk':>6} {'A*':>6} {'Saved':>6} {'Same?':>6}")
    print(f"  {'-'*42} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for src, tgt, label in test_pairs:
        cmp  = compare_algorithms(graph, src, tgt, "morning")
        d, a = cmp['dijkstra'], cmp['astar']
        same = "✅" if cmp['same_path'] else "❌"
        print(f"  {label:<42} {d.nodes_explored:>6} {a.nodes_explored:>6} "
              f"{cmp['nodes_saved']:>6} {same:>6}")


# ═════════════════════════════════════════════════════════════════
# MEMBER 5 — TRANSIT DP
# ═════════════════════════════════════════════════════════════════

def demo_transit(save_png: bool = True):
    separator("MEMBER 5 — Public Transit Scheduler  |  DP Bus Allocation")
    result = optimize_transit()
    result.summary()
    if save_png:
        fig = plot_transit(result)
        out = p("transit_allocation.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  📊 Chart saved → {out}")
    return result


# ═════════════════════════════════════════════════════════════════
# MEMBER 6 — MAINTENANCE DP
# ═════════════════════════════════════════════════════════════════

def demo_maintenance(save_png: bool = True):
    separator("MEMBER 6 — Infrastructure Maintenance  |  DP Knapsack")
    edges  = load_edges(p("edges.csv"))
    nodes  = load_nodes(p("nodes.csv"))
    result = solve_maintenance(edges, budget=2000)
    result.summary()
    if save_png:
        fig = plot_condition_heatmap(edges, nodes, result)
        out = p("maintenance_heatmap.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  📊 Heatmap saved → {out}")
    return result


# ═════════════════════════════════════════════════════════════════
# MEMBER 7 — GREEDY SIGNALS
# ═════════════════════════════════════════════════════════════════

def demo_greedy(graph: CairoGraph, save_png: bool = True):
    separator("MEMBER 7 — Traffic Signal Optimizer  |  Greedy Algorithm")

    intersections = [
        ("3",  "morning"),    # Downtown Cairo — busiest hub
        ("9",  "evening"),    # Mohandessin — evening peak
        ("11", "morning"),    # Shubra — morning peak
    ]

    for node_id, slot in intersections:
        controller = SignalController(graph, intersection_id=node_id, time_of_day=slot)
        result     = controller.run_simulation(steps=10)
        result.summary()

        if save_png:
            fig = plot_signal_dashboard(result)
            out = p(f"signal_dashboard_{node_id}_{slot}.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  📊 Dashboard saved → {out}")


# ═════════════════════════════════════════════════════════════════
# FULL SUMMARY
# ═════════════════════════════════════════════════════════════════

def demo_full_summary(graph: CairoGraph):
    separator("SYSTEM SUMMARY  |  Cairo Smart City Transportation Network")
    stats = graph.summary()
    print(f"\n  Nodes           : {stats['total_nodes']}")
    print(f"  Existing roads  : {stats['existing_roads']}")
    print(f"  Potential roads : {stats['potential_roads']}")
    print(f"  ML active       : {'Yes' if stats['ml_active'] else 'No (distance fallback)'}")
    print(f"  Medical fac.    : {', '.join(stats['medical_facilities'])}")
    print(f"  Gov. centres    : {', '.join(stats['gov_centers'])}")
    print(f"\n  Top-5 populated areas:")
    for n in graph.get_high_population_nodes(5):
        print(f"    {n.name:<35} {int(n.population):>8,}")


# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cairo Smart City — CSE112 Project Demo"
    )
    parser.add_argument("--module", type=int, choices=[2, 3, 4, 5, 6, 7],
        help="Run only one member's demo")
    parser.add_argument("--live",   action="store_true",
        help="Fetch live weights from TomTom API (requires TOMTOM_API_KEY)")
    parser.add_argument("--quiet",  action="store_true",
        help="Suppress graph-loading output")
    parser.add_argument("--no-png", action="store_true",
        help="Skip saving PNG charts")
    args = parser.parse_args()

    print("\n🚀  Cairo Smart City Transportation Network — CSE112")
    graph = build_graph(quiet=args.quiet, use_live=args.live)

    save = not args.no_png

    if   args.module == 2: demo_mst(graph, save)
    elif args.module == 3: demo_dijkstra(graph)
    elif args.module == 4: demo_astar(graph)
    elif args.module == 5: demo_transit(save)
    elif args.module == 6: demo_maintenance(save)
    elif args.module == 7: demo_greedy(graph, save)
    else:
        demo_full_summary(graph)
        demo_mst(graph, save)
        demo_dijkstra(graph)
        demo_astar(graph)
        demo_transit(save)
        demo_maintenance(save)
        demo_greedy(graph, save)

    print("\n✅  Demo complete.\n")


if __name__ == "__main__":
    main()
