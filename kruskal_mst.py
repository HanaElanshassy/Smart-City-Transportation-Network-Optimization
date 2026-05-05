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
    # result['chosen_roads']  -> list of Edge objects selected by MST
    # result['total_cost']    -> float, total construction cost in Million EGP
    # result['savings']       -> float, cost saved via priority discounts

    fig = visualize_infrastructure(graph, result['chosen_roads'])
    fig.savefig('mst_map.png', dpi=150, bbox_inches='tight')
"""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers and Vercel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from data_loader1 import CairoGraph

# Nodes that receive a 50% construction-cost discount to ensure
# Kruskal's prioritises connecting them early.
PRIORITY_NODES = {'8', '2', 'F9'}   # Giza, Nasr City, Qasr El Aini Hospital
PRIORITY_DISCOUNT = 0.5


# ─────────────────────────────────────────────
# UNION-FIND
# ─────────────────────────────────────────────

class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank   = {el: 0  for el in elements}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # path compression
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


# ─────────────────────────────────────────────
# KRUSKAL'S ALGORITHM
# ─────────────────────────────────────────────

def plan_new_infrastructure(graph: CairoGraph) -> dict:
    """
    Run Kruskal's MST on potential new roads to find the most cost-effective
    expansion plan, with priority discounts for Giza, Nasr City, and F9.

    Args:
        graph : loaded CairoGraph instance

    Returns:
        dict with keys:
            'chosen_roads' : list[Edge]  — edges selected by MST
            'total_cost'   : float       — actual construction cost (Million EGP)
            'savings'      : float       — cost saved from priority discounts
    """
    uf = UnionFind(graph.nodes.keys())

    # Pre-connect all nodes that already have existing roads
    for edge in graph.get_existing_edges():
        uf.union(edge.from_id, edge.to_id)

    # Apply priority discount and sort by modified cost
    weighted = []
    for edge in graph.get_potential_edges():
        is_priority = edge.from_id in PRIORITY_NODES or edge.to_id in PRIORITY_NODES
        discount    = PRIORITY_DISCOUNT if is_priority else 1.0
        weighted.append((edge.cost * discount, edge.cost, edge))
    weighted.sort(key=lambda x: x[0])

    chosen_roads = []
    total_cost   = 0.0
    savings      = 0.0

    for modified_cost, actual_cost, edge in weighted:
        if uf.union(edge.from_id, edge.to_id):
            chosen_roads.append(edge)
            total_cost += actual_cost
            savings    += actual_cost - modified_cost

    return {
        'chosen_roads': chosen_roads,
        'total_cost':   total_cost,
        'savings':      savings,
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize_infrastructure(graph: CairoGraph, chosen_roads: list) -> plt.Figure:
    """
    Generate a matplotlib map of Cairo's road network showing existing roads
    and the MST-selected new roads.

    Args:
        graph         : loaded CairoGraph instance
        chosen_roads  : list of Edge objects returned by plan_new_infrastructure()

    Returns:
        matplotlib Figure (caller is responsible for saving or displaying it)
    """
    G   = nx.Graph()
    pos = {}

    # Node colour scheme
    node_colors = []
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.name)
        pos[node_id] = (node.x, node.y)

        if node.is_medical:
            node_colors.append('#e74c3c')       # red  — medical facilities
        elif node_id in PRIORITY_NODES:
            node_colors.append('#f1c40f')       # gold — priority targets
        elif node.is_facility:
            node_colors.append('#e67e22')       # orange — other facilities
        else:
            node_colors.append('#aed6f1')       # blue  — regular neighbourhoods

    existing_edgelist = [(e.from_id, e.to_id) for e in graph.get_existing_edges()]
    new_edgelist      = [(e.from_id, e.to_id) for e in chosen_roads]

    G.add_edges_from(existing_edgelist)
    G.add_edges_from(new_edgelist)

    fig, ax = plt.subplots(figsize=(16, 11))

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors,
                           edgecolors='black', linewidths=0.8, ax=ax)

    # Existing roads
    nx.draw_networkx_edges(G, pos, edgelist=existing_edgelist,
                           width=1.5, edge_color='#7f8c8d', alpha=0.6, ax=ax)

    # New MST roads
    nx.draw_networkx_edges(G, pos, edgelist=new_edgelist,
                           width=3.0, edge_color='#27ae60', style='dashed', ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G, 'label'),
                            font_size=7, font_weight='bold', ax=ax)

    # Legend
    legend_handles = [
        mpatches.Patch(color='#e74c3c', label='Medical Facility'),
        mpatches.Patch(color='#f1c40f', label='Priority Node (Giza / Nasr City / F9)'),
        mpatches.Patch(color='#e67e22', label='Other Facility'),
        mpatches.Patch(color='#aed6f1', label='Neighbourhood'),
        mpatches.Patch(color='#7f8c8d', label='Existing Road'),
        mpatches.Patch(color='#27ae60', label='Proposed New Road (MST)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=9)
    ax.set_title("Cairo Smart City — Infrastructure Expansion Plan (Kruskal's MST)", fontsize=15)
    ax.axis('off')
    fig.tight_layout()
    return fig
    