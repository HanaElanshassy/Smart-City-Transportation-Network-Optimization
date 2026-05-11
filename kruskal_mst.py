"""
CSE112 - Member 2: Infrastructure Designer (MST)
Cairo Smart City Transportation Network Optimization

Uses Kruskal's Algorithm to select the most cost-effective new roads
from the potential edges dataset, prioritizing connections to Giza (8),
Nasr City (2), and Qasr El Aini Hospital (F9) via a 50% cost discount.

Public API (used by the main app / UI):
    from kruskal_mst import plan_new_infrastructure, visualize_infrastructure

    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')

    result = plan_new_infrastructure(graph)
    # result['chosen_roads'] -> list of Edge objects selected by MST
    # result['total_cost']   -> float, total construction cost in Million EGP
    # result['savings']      -> float, cost saved via priority discounts

    fig = visualize_infrastructure(graph, result['chosen_roads'])
    fig.write_html('mst_map.html')
"""

from data_loader1 import CairoGraph


# Nodes that receive a 50% construction-cost discount to ensure
# Kruskal's prioritizes connecting them early.
PRIORITY_NODES = {"8", "2", "F9"}  # Giza, Nasr City, Qasr El Aini Hospital
PRIORITY_DISCOUNT = 0.5


# UNION-FIND

class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# KRUSKAL'S ALGORITHM

def plan_new_infrastructure(graph: CairoGraph) -> dict:
    """
    Run Kruskal's MST on potential new roads to find the most cost-effective
    expansion plan, with priority discounts for Giza, Nasr City, and F9.

    Complexity analysis:
        Time Complexity: O(E log E), equivalently O(E log V) for a simple graph.
            Building the candidate edge list takes O(E). Sorting those edges by
            adjusted construction cost dominates at O(E log E). The union-find
            pass performs find/union with path compression and union by rank, so
            processing all edges is O(E alpha(V)), effectively near constant per
            operation.

        Space Complexity: O(V + E).
            The parent and rank dictionaries store one entry per node, requiring
            O(V) space. The weighted candidate list and seen_candidates set scale
            with the number of edges, requiring O(E) space. chosen_roads stores
            at most V - 1 selected roads, so the total memory footprint is
            O(V + E).

    Args:
        graph : loaded CairoGraph instance

    Returns:
        dict with keys:
            'chosen_roads' : list[Edge] - edges selected by MST
            'total_cost'   : float      - actual construction cost (Million EGP)
            'savings'      : float      - cost saved from priority discounts
    """
    uf = UnionFind(graph.nodes.keys())

    # Pre-connect the base road network. Facility connectors with construction
    # costs are usable for routing, but still remain candidates for this
    # infrastructure-planning demo.
    for edge in graph.get_existing_edges():
        if edge.cost == 0:
            uf.union(edge.from_id, edge.to_id)

    weighted = []
    seen_candidates = set()
    for edge in graph.edges:
        if edge.cost <= 0:
            continue
        key = tuple(sorted([edge.from_id, edge.to_id]))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        is_priority = edge.from_id in PRIORITY_NODES or edge.to_id in PRIORITY_NODES
        discount = PRIORITY_DISCOUNT if is_priority else 1.0
        weighted.append((edge.cost * discount, edge.cost, edge))
    weighted.sort(key=lambda x: x[0])

    chosen_roads = []
    total_cost = 0.0
    savings = 0.0

    for modified_cost, actual_cost, edge in weighted:
        if uf.union(edge.from_id, edge.to_id):
            chosen_roads.append(edge)
            total_cost += actual_cost
            savings += actual_cost - modified_cost

    return {
        "chosen_roads": chosen_roads,
        "total_cost": total_cost,
        "savings": savings,
    }


# VISUALIZATION

def visualize_infrastructure(graph: CairoGraph, chosen_roads: list):
    """
    Generate a real Plotly Mapbox Cairo map showing existing roads and the
    MST-selected new roads.

    Args:
        graph         : loaded CairoGraph instance
        chosen_roads  : list of Edge objects returned by plan_new_infrastructure()

    Returns:
        Plotly Figure (caller is responsible for displaying or saving it)
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    chosen_keys = {
        tuple(sorted([edge.from_id, edge.to_id]))
        for edge in chosen_roads
    }

    def add_road_trace(edge, color, width, opacity=1.0, name=None, showlegend=False):
        n_from = graph.get_node(edge.from_id)
        n_to = graph.get_node(edge.to_id)
        if not n_from or not n_to:
            return

        line = dict(color=color, width=width)

        fig.add_trace(go.Scattermapbox(
            lon=[n_from.x, n_to.x],
            lat=[n_from.y, n_to.y],
            mode="lines",
            line=line,
            opacity=opacity,
            hoverinfo="text",
            hovertext=(
                f"{n_from.name} -> {n_to.name}<br>"
                f"Distance: {edge.distance} km<br>"
                f"Condition: {edge.condition}/10"
                + (f"<br>Cost: {edge.cost:.0f}M EGP" if edge.cost > 0 else "")
            ),
            name=name,
            showlegend=showlegend,
        ))

    for edge in graph.get_existing_edges():
        add_road_trace(edge, "#4a7a9b", 2.0)

    for edge in graph.get_potential_edges():
        key = tuple(sorted([edge.from_id, edge.to_id]))
        if key not in chosen_keys:
            add_road_trace(edge, "#2244aa", 1.2, opacity=0.35)

    for i, edge in enumerate(chosen_roads):
        add_road_trace(
            edge,
            "#00ff88",
            4.0,
            opacity=1.0,
            name="Proposed New Road (MST)" if i == 0 else None,
            showlegend=(i == 0),
        )

    node_palette = {
        "Medical": "#ff3366",
        "Government": "#aa44ff",
        "Business": "#00aaff",
        "Mixed": "#00ccaa",
        "Residential": "#4488cc",
        "Industrial": "#ff8800",
        "Airport": "#ffdd00",
        "Transit Hub": "#ff8844",
        "Education": "#44ddff",
        "Tourism": "#ff44aa",
        "Sports": "#44ff88",
        "Commercial": "#ffaa00",
    }

    by_type = {}
    for node_id, node in graph.nodes.items():
        node_type = node.type
        by_type.setdefault(
            node_type,
            {"lons": [], "lats": [], "texts": [], "labels": [], "sizes": []},
        )
        priority = node_id in PRIORITY_NODES
        by_type[node_type]["lons"].append(node.x)
        by_type[node_type]["lats"].append(node.y)
        by_type[node_type]["labels"].append(node.name)
        by_type[node_type]["sizes"].append(14 if priority or node.is_medical else 9)
        by_type[node_type]["texts"].append(
            f"<b>{node.name}</b><br>"
            f"ID: {node.id}<br>"
            f"Type: {node.type}"
            + (f"<br>Population: {int(node.population):,}" if node.population > 0 else "")
            + ("<br>Priority target" if priority else "")
        )

    for node_type, data in by_type.items():
        fig.add_trace(go.Scattermapbox(
            lon=data["lons"],
            lat=data["lats"],
            mode="markers+text",
            marker=dict(
                size=data["sizes"],
                color=node_palette.get(node_type, "#aaccee"),
                opacity=0.96,
                symbol="circle",
            ),
            text=data["labels"],
            textposition="top right",
            textfont=dict(color="#c8e0f4", size=8),
            hovertext=data["texts"],
            hoverinfo="text",
            name=node_type,
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=30.03, lon=31.25),
            zoom=10.3,
        ),
        paper_bgcolor="#060b14",
        plot_bgcolor="#060b14",
        margin=dict(l=0, r=0, t=30, b=0),
        height=580,
        title=dict(
            text="Cairo Smart City - Infrastructure Expansion Plan (Kruskal's MST)",
            font=dict(color="#e8f4ff", size=16),
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
