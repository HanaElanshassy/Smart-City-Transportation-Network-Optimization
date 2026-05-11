"""
CSE112 - Member 6: Infrastructure Maintenance (DP Knapsack)
Cairo Smart City Transportation Network Optimization

Maximizes the total network condition-score improvement within a repair
budget using a 0/1 Knapsack DP approach.
"""

import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def load_edges(edges_path: str) -> list:
    """
    Parse edges.csv and derive maintenance cost and value for each road.

    repair_cost = distance * (11 - condition) * 10
    repair_value = (10 - condition) * (capacity // 500)
    """
    edges = []
    with open(edges_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = int(row["Condition"])
            if condition <= 0:
                continue

            from_id = row["FromID"]
            to_id = row["ToID"]
            distance = float(row["Distance"])
            capacity = int(row["Capacity"])

            repair_cost = int(distance * (11 - condition) * 10)
            repair_value = (10 - condition) * (capacity // 500)

            edges.append({
                "id": f"{from_id}-{to_id}",
                "from_id": from_id,
                "to_id": to_id,
                "distance": distance,
                "capacity": capacity,
                "condition": condition,
                "repair_cost": repair_cost,
                "repair_value": repair_value,
            })
    return edges


def load_nodes(nodes_path: str) -> dict:
    """Parse nodes.csv and return {node_id: {name, x, y}}."""
    nodes = {}
    with open(nodes_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes[row["ID"]] = {
                "name": row["Name"],
                "x": float(row["X"]),
                "y": float(row["Y"]),
            }
    return nodes


class MaintenanceResult:
    def __init__(self, budget, selected_roads, total_cost, total_value, all_edges):
        self.budget = budget
        self.selected_roads = selected_roads
        self.total_cost = total_cost
        self.total_value = total_value
        self.all_edges = all_edges

    def summary(self):
        print("\n" + "=" * 62)
        print("  INFRASTRUCTURE MAINTENANCE OPTIMIZER  (DP Knapsack)")
        print("=" * 62)
        print(f"  Budget Available  : {self.budget:,} cost-units")
        print(f"  Budget Used       : {self.total_cost:,} cost-units")
        print(f"  Budget Remaining  : {self.budget - self.total_cost:,} cost-units")
        print(f"  Score Gain        : +{self.total_value} condition points")
        print(f"  Roads Selected    : {len(self.selected_roads)}")
        print("-" * 62)
        print(f"  {'Road':<15} {'Condition':>12} {'Cost':>10} {'Value':>7}")
        print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*7}")
        for road in sorted(self.selected_roads, key=lambda x: x["condition"]):
            if road["condition"] <= 4:
                label = "Poor"
            elif road["condition"] <= 7:
                label = "Fair"
            else:
                label = "Good"
            print(
                f"  {road['id']:<15} {label:>12}  {road['condition']}/10  "
                f"{road['repair_cost']:>8,}   {road['repair_value']:>5}"
            )
        print("=" * 62)


def solve_maintenance(edges: list, budget: int) -> MaintenanceResult:
    """
    0/1 Knapsack DP to choose which roads to repair.

    Time complexity: O(n * W), where W is the budget.
    Space complexity: O(n * W).
    """
    candidates = [
        edge for edge in edges
        if edge["repair_value"] > 0 and edge["repair_cost"] <= budget
    ]

    n = len(candidates)
    W = budget
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost = candidates[i - 1]["repair_cost"]
        value = candidates[i - 1]["repair_value"]
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= cost:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - cost] + value)

    selected = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(candidates[i - 1])
            w -= candidates[i - 1]["repair_cost"]

    total_cost = sum(road["repair_cost"] for road in selected)
    total_value = sum(road["repair_value"] for road in selected)

    return MaintenanceResult(budget, selected, total_cost, total_value, edges)


def plot_condition_heatmap(edges: list, nodes: dict, result: MaintenanceResult):
    """
    Draw a real Plotly Mapbox Cairo map colored by road condition.

    DP-selected roads for repair are highlighted with thick bright blue lines.
    Returns a Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    selected_ids = {road["id"] for road in result.selected_roads}
    selected_ids |= {
        "-".join(reversed(road["id"].split("-")))
        for road in result.selected_roads
    }

    def condition_style(condition: int):
        if condition <= 4:
            return "#ff3366", "Poor Condition (1-4)"
        if condition <= 7:
            return "#ffcc00", "Fair Condition (5-7)"
        return "#00cc66", "Good Condition (8-10)"

    legend_seen = set()

    for edge in edges:
        n_from = nodes.get(edge["from_id"])
        n_to = nodes.get(edge["to_id"])
        if not n_from or not n_to:
            continue

        base_color, condition_label = condition_style(edge["condition"])
        is_selected = edge["id"] in selected_ids
        color = "#00d4ff" if is_selected else base_color
        width = 5.0 if is_selected else 2.4
        opacity = 1.0 if is_selected else 0.72
        trace_name = (
            f"Selected for Repair - DP ({len(result.selected_roads)} roads)"
            if is_selected else condition_label
        )
        legend_key = "selected" if is_selected else condition_label
        showlegend = legend_key not in legend_seen
        legend_seen.add(legend_key)

        fig.add_trace(go.Scattermapbox(
            lon=[n_from["x"], n_to["x"]],
            lat=[n_from["y"], n_to["y"]],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            hoverinfo="text",
            hovertext=(
                f"{n_from['name']} -> {n_to['name']}<br>"
                f"Condition: {edge['condition']}/10<br>"
                f"Distance: {edge['distance']} km<br>"
                f"Repair Cost: {edge['repair_cost']:,} units<br>"
                f"Score Gain: +{edge['repair_value']}"
                + ("<br><b>Selected for repair</b>" if is_selected else "")
            ),
            name=trace_name,
            showlegend=showlegend,
        ))

    fig.add_trace(go.Scattermapbox(
        lon=[node["x"] for node in nodes.values()],
        lat=[node["y"] for node in nodes.values()],
        mode="markers+text",
        marker=dict(size=9, color="#d5e8f7", opacity=0.95, symbol="circle"),
        text=[node["name"] for node in nodes.values()],
        textposition="top right",
        textfont=dict(color="#c8e0f4", size=8),
        hovertext=[f"<b>{node['name']}</b>" for node in nodes.values()],
        hoverinfo="text",
        name="Network Nodes",
        showlegend=True,
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=30.03, lon=31.25),
            zoom=10.3,
        ),
        paper_bgcolor="#060b14",
        plot_bgcolor="#060b14",
        margin=dict(l=0, r=0, t=42, b=0),
        height=580,
        title=dict(
            text=(
                "Cairo Smart City - Infrastructure Maintenance Map"
                f" | Budget: {result.budget:,} | Used: {result.total_cost:,}"
                f" | Score Gain: +{result.total_value}"
            ),
            font=dict(color="#e8f4ff", size=15),
            x=0.02,
            y=0.98,
        ),
        legend=dict(
            bgcolor="rgba(10, 22, 40, 0.82)",
            bordercolor="#1a3a5c",
            borderwidth=1,
            font=dict(color="#8ba8cc", size=10),
            x=0.01,
            y=0.98,
        ),
        hoverlabel=dict(
            bgcolor="#0d1e35",
            bordercolor="#1a3a5c",
            font=dict(color="#e0f0ff", size=11),
        ),
    )
    return fig


if __name__ == "__main__":
    def p(filename):
        return os.path.join(BASE_DIR, filename)

    try:
        edges = load_edges(p("edges.csv"))
        nodes = load_nodes(p("nodes.csv"))
        print(f"Loaded {len(edges)} roads.")

        result = solve_maintenance(edges, 2000)
        result.summary()

        fig = plot_condition_heatmap(edges, nodes, result)
        out = p("maintenance_heatmap.html")
        fig.write_html(out)
        print(f"\n  Map saved -> {out}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}. Ensure edges.csv and nodes.csv are present.")
