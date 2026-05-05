"""
=============================================================
  CSE112 - Member 6: Infrastructure Maintenance (DP Knapsack)
  Cairo Smart City Transportation Network Optimization
=============================================================

Maximizes the total network "Condition Score" improvement
within a given Repair Budget using a 0/1 Knapsack DP approach.

Fixes vs. original:
  - plt.show() replaced with savefig() for server/Vercel safety
  - Arabic comments translated to English for team consistency
  - plot_condition_heatmap now returns the Figure and saves to disk
  - repair_value formula documented clearly
  - budget label now explicit: budget is in the same unit as
    repair_cost (distance × condition-gap × 10, dimensionless
    "cost units") — kept as-is since the scale is self-consistent

Public API:
    edges  = load_edges('edges.csv')
    nodes  = load_nodes('nodes.csv')
    result = solve_maintenance(edges, budget=2000)
    result.summary()
    fig    = plot_condition_heatmap(edges, nodes, result)
    fig.savefig('maintenance_heatmap.png', ...)
"""

import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')   # non-interactive — safe for servers / Vercel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ─────────────────────────────────────────────
# 1. LOAD DATA DIRECTLY FROM CSV
# ─────────────────────────────────────────────

def load_edges(edges_path: str) -> list:
    """
    Parse edges.csv and derive maintenance cost & value for each road.

    repair_cost  = distance × (11 - condition) × 10
        Longer roads in worse condition cost more to repair.

    repair_value = (10 - condition) × (capacity // 500)
        Roads in worse condition serving more traffic yield
        higher benefit when repaired.
        Roads already at condition 10 have repair_value = 0
        and are excluded from the knapsack candidates.
    """
    edges = []
    with open(edges_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = int(row['Condition'])
            if condition <= 0:
                continue                    # skip invalid rows

            from_id  = row['FromID']
            to_id    = row['ToID']
            distance = float(row['Distance'])
            capacity = int(row['Capacity'])

            repair_cost  = int(distance * (11 - condition) * 10)
            repair_value = (10 - condition) * (capacity // 500)

            edges.append({
                'id':           f"{from_id}-{to_id}",
                'from_id':      from_id,
                'to_id':        to_id,
                'distance':     distance,
                'capacity':     capacity,
                'condition':    condition,
                'repair_cost':  repair_cost,
                'repair_value': repair_value,
            })
    return edges


def load_nodes(nodes_path: str) -> dict:
    """Parse nodes.csv and return {node_id: {name, x, y}}."""
    nodes = {}
    with open(nodes_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes[row['ID']] = {
                'name': row['Name'],
                'x':    float(row['X']),
                'y':    float(row['Y']),
            }
    return nodes


# ─────────────────────────────────────────────
# 2. RESULT CLASS
# ─────────────────────────────────────────────

class MaintenanceResult:
    def __init__(self, budget, selected_roads, total_cost, total_value, all_edges):
        self.budget         = budget
        self.selected_roads = selected_roads
        self.total_cost     = total_cost
        self.total_value    = total_value
        self.all_edges      = all_edges

    def summary(self):
        print("\n" + "═" * 62)
        print("  INFRASTRUCTURE MAINTENANCE OPTIMIZER  (DP Knapsack)")
        print("═" * 62)
        print(f"  Budget Available  : {self.budget:,} cost-units")
        print(f"  Budget Used       : {self.total_cost:,} cost-units")
        print(f"  Budget Remaining  : {self.budget - self.total_cost:,} cost-units")
        print(f"  Score Gain        : +{self.total_value} condition points")
        print(f"  Roads Selected    : {len(self.selected_roads)}")
        print("─" * 62)
        print(f"  {'Road':<15} {'Condition':>12} {'Cost':>10} {'Value':>7}")
        print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*7}")
        for r in sorted(self.selected_roads, key=lambda x: x['condition']):
            if r['condition'] <= 4:
                icon = "🔴 Poor"
            elif r['condition'] <= 7:
                icon = "🟡 Fair"
            else:
                icon = "🟢 Good"
            print(f"  {r['id']:<15} {icon:>12}  {r['condition']}/10  "
                  f"{r['repair_cost']:>8,}   {r['repair_value']:>5}")
        print("═" * 62)


# ─────────────────────────────────────────────
# 3. DP KNAPSACK SOLVER
# ─────────────────────────────────────────────

def solve_maintenance(edges: list, budget: int) -> MaintenanceResult:
    """
    0/1 Knapsack DP to choose which roads to repair.

    Only roads with repair_value > 0 (condition < 10) and
    repair_cost ≤ budget are considered.

    Time  complexity: O(n × W)  where W = budget
    Space complexity: O(n × W)

    Args:
        edges  : list from load_edges()
        budget : integer repair budget in cost-units

    Returns:
        MaintenanceResult with chosen roads and metrics
    """
    candidates = [
        e for e in edges
        if e['repair_value'] > 0 and e['repair_cost'] <= budget
    ]

    n = len(candidates)
    W = budget

    # Build DP table: dp[i][w] = max value using first i items with weight ≤ w
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost  = candidates[i - 1]['repair_cost']
        value = candidates[i - 1]['repair_value']
        for w in range(W + 1):
            # Option A: skip road i
            dp[i][w] = dp[i - 1][w]
            # Option B: repair road i (only if budget allows)
            if w >= cost:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - cost] + value)

    # Backtrack to find selected roads
    selected = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(candidates[i - 1])
            w -= candidates[i - 1]['repair_cost']

    total_cost  = sum(r['repair_cost']  for r in selected)
    total_value = sum(r['repair_value'] for r in selected)

    return MaintenanceResult(budget, selected, total_cost, total_value, edges)


# ─────────────────────────────────────────────
# 4. CONDITION HEATMAP VISUALIZATION
# ─────────────────────────────────────────────

def plot_condition_heatmap(
    edges: list,
    nodes: dict,
    result: MaintenanceResult,
) -> plt.Figure:
    """
    Draw a geographical network map colour-coded by road condition:
        Red  (🔴) — Poor condition  (1–4)
        Gold (🟡) — Fair condition  (5–7)
        Green(🟢) — Good condition  (8–10)

    DP-selected roads for repair are highlighted with thick dashed blue edges.

    Returns a matplotlib Figure (caller saves or displays it).
    """
    G   = nx.Graph()
    pos = {}

    for node_id, data in nodes.items():
        G.add_node(node_id, label=data['name'])
        pos[node_id] = (data['x'], data['y'])

    poor_edges, fair_edges, good_edges = [], [], []
    selected_ids = {r['id'] for r in result.selected_roads}
    # Build reverse-key set too (e.g. "3-1" for "1-3")
    selected_ids |= {'-'.join(reversed(r['id'].split('-'))) for r in result.selected_roads}

    selected_edge_list = []

    for e in edges:
        u, v = e['from_id'], e['to_id']
        G.add_edge(u, v)
        edge_tuple = (u, v)

        if e['id'] in selected_ids:
            selected_edge_list.append(edge_tuple)

        if e['condition'] <= 4:
            poor_edges.append(edge_tuple)
        elif e['condition'] <= 7:
            fair_edges.append(edge_tuple)
        else:
            good_edges.append(edge_tuple)

    fig, ax = plt.subplots(figsize=(15, 10))

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=420, node_color='#d5d8dc',
                           edgecolors='black', linewidths=0.8, ax=ax)

    # Edges by condition
    nx.draw_networkx_edges(G, pos, edgelist=poor_edges,
                           width=2.0, edge_color='#e74c3c', alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fair_edges,
                           width=2.0, edge_color='#f1c40f', alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=good_edges,
                           width=2.0, edge_color='#27ae60', alpha=0.85, ax=ax)

    # DP-selected roads — thick dashed blue
    if selected_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=selected_edge_list,
                               width=4.5, edge_color='#2980b9',
                               style='dashed', ax=ax)

    # Labels
    nx.draw_networkx_labels(
        G, pos, nx.get_node_attributes(G, 'label'),
        font_size=7, font_weight='bold', ax=ax
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color='#e74c3c', label='Poor Condition  (1–4)'),
        mpatches.Patch(color='#f1c40f', label='Fair Condition  (5–7)'),
        mpatches.Patch(color='#27ae60', label='Good Condition  (8–10)'),
        mpatches.Patch(color='#2980b9', label=f'Selected for Repair — DP  ({len(result.selected_roads)} roads)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=9)
    ax.set_title(
        f"Cairo Smart City — Infrastructure Maintenance Heatmap\n"
        f"Budget: {result.budget:,} cost-units  |  "
        f"Used: {result.total_cost:,}  |  Score Gain: +{result.total_value}",
        fontsize=14
    )
    ax.axis('off')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    def p(f): return os.path.join(BASE_DIR, f)

    try:
        edges = load_edges(p("edges.csv"))
        nodes = load_nodes(p("nodes.csv"))
        print(f"✅ Loaded {len(edges)} roads.")

        BUDGET = 2000
        result = solve_maintenance(edges, BUDGET)
        result.summary()

        fig = plot_condition_heatmap(edges, nodes, result)
        out = p("maintenance_heatmap.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  📊 Heatmap saved → {out}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}. Ensure edges.csv and nodes.csv are present.")