"""
=============================================================
  CSE112 - Member 2: Infrastructure Designer (MST)
  Cairo Smart City Transportation Network Optimization
=============================================================
Uses Kruskal's Algorithm to find the Minimum Spanning Tree (MST) 
for new potential roads, prioritizing Giza, Nasr City, and Qasr El Aini.
Visualizes the result using NetworkX and Matplotlib.
"""

import matplotlib.pyplot as plt
import networkx as nx
from data_loader import CairoGraph

# ─────────────────────────────────────────────
# 1. UNION-FIND DATA STRUCTURE
# ─────────────────────────────────────────────
class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, item):
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False

# ─────────────────────────────────────────────
# 2. KRUSKAL'S ALGORITHM
# ─────────────────────────────────────────────
def plan_new_infrastructure(graph):
    uf = UnionFind(graph.nodes.keys())
    existing_edges = graph.get_existing_edges()
    potential_edges = graph.get_potential_edges()

    # Step 1: Connect all nodes that already have existing roads
    for edge in existing_edges:
        uf.union(edge.from_id, edge.to_id)

    # Step 2: Prioritize specific nodes (Giza: '8', Nasr City: '2', Qasr El Aini: 'F9')
    priority_nodes = {'8', '2', 'F9'}
    
    modified_potential_edges = []
    for edge in potential_edges:
        # If the road connects to a priority node, artificially reduce its cost by 50%
        # so Kruskal's algorithm picks it first.
        discount_factor = 0.5 if (edge.from_id in priority_nodes or edge.to_id in priority_nodes) else 1.0
        modified_cost = edge.cost * discount_factor
        modified_potential_edges.append((modified_cost, edge))

    # Sort by the modified (discounted) cost
    modified_potential_edges.sort(key=lambda x: x[0])

    # Step 3: Run Kruskal's to pick the best new roads
    chosen_new_roads = []
    total_budget = 0.0

    for modified_cost, edge in modified_potential_edges:
        if uf.union(edge.from_id, edge.to_id):
            chosen_new_roads.append(edge)
            total_budget += edge.cost  # Add actual cost to budget

    return chosen_new_roads, total_budget

# ─────────────────────────────────────────────
# 3. VISUALIZATION (MAP PLOT)
# ─────────────────────────────────────────────
def visualize_infrastructure(graph, chosen_new_roads):
    print("🎨 Generating Network Visualization...")
    G = nx.Graph()

    # Add Nodes with coordinates
    pos = {}
    node_colors = []
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.name)
        pos[node_id] = (node.x, node.y)
        
        if node.is_medical:
            node_colors.append('red')
        elif node_id in ['8', '2', 'F9']: # Priority targets
            node_colors.append('gold')
        else:
            node_colors.append('lightblue')

    # Add Existing Roads (Grey)
    existing_edges = [(e.from_id, e.to_id) for e in graph.get_existing_edges()]
    G.add_edges_from(existing_edges)

    # Add Chosen New Roads (Green & Bold)
    new_edges = [(e.from_id, e.to_id) for e in chosen_new_roads]
    G.add_edges_from(new_edges)

    plt.figure(figsize=(14, 10))
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors='black')
    
    # Draw Existing Roads
    nx.draw_networkx_edges(G, pos, edgelist=existing_edges, width=1.5, edge_color='gray', alpha=0.5, label='Existing Roads')
    
    # Draw Chosen New Roads
    nx.draw_networkx_edges(G, pos, edgelist=new_edges, width=3.0, edge_color='green', style='dashed', label='Proposed New Roads')
    
    # Draw Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title("Cairo Smart City: Infrastructure Expansion Plan (MST)", fontsize=16)
    plt.legend(["Medical/Priority", "Normal Node", "Existing Roads", "Proposed Roads (MST)"], loc="best")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CSE112 - Infrastructure Expansion Planning (Kruskal's)")
    print("=" * 60)
    
    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')
    
    chosen_roads, total_budget = plan_new_infrastructure(graph)
    
    print(f"\n✅ Infrastructure Plan Generated!")
    print(f"💰 Total Estimated Budget for New Roads: {total_budget:.2f} Million EGP\n")
    print("🛣️  Approved New Roads to Build:")
    for road in chosen_roads:
        print(f"   ➜ Connect {graph.get_node(road.from_id).name} with {graph.get_node(road.to_id).name} (Cost: {road.cost}M)")
    
    visualize_infrastructure(graph, chosen_roads)