"""
=============================================================
  CSE112 - Member 6: Infrastructure Maintenance (DP Knapsack)
  Cairo Smart City Transportation Network Optimization
=============================================================

Maximizes the total network "Condition Score" within a given
Repair Budget using a 0/1 Knapsack Dynamic Programming approach.
Visualizes the result using a Geographical Heatmap (X/Y).
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────
# 1. LOAD DATA DIRECTLY FROM CSV
# ─────────────────────────────────────────────

def load_edges(edges_path):
    edges = []
    with open(edges_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = int(row['Condition'])
            cost_raw  = float(row['Cost'])
            
            # نتعامل فقط مع الطرق الحالية (اللي تكلفتها 0 وحالتها أكبر من 0)
            if condition > 0:
                from_id  = row['FromID']
                to_id    = row['ToID']
                distance = float(row['Distance'])
                capacity = int(row['Capacity'])

                # تكلفة الإصلاح: بتزيد كل ما الطريق كان أطول وحالته أسوأ
                repair_cost = int(distance * (11 - condition) * 10)

                # قيمة الإصلاح (المنفعة): بتزيد لو الطريق حيوي (Capacity عالية) وحالته سيئة
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

def load_nodes(nodes_path):
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
# 2. RESULT DATACLASS
# ─────────────────────────────────────────────

class MaintenanceResult:
    def __init__(self, budget, selected_roads, total_cost, total_value, all_edges):
        self.budget         = budget
        self.selected_roads = selected_roads
        self.total_cost     = total_cost
        self.total_value    = total_value
        self.all_edges      = all_edges

    def summary(self):
        print("\n" + "═" * 60)
        print("  🏗️   INFRASTRUCTURE MAINTENANCE OPTIMIZER (DP)")
        print("═" * 60)
        print(f"  💰 Budget Available : {self.budget:,} EGP")
        print(f"  📉 Budget Used      : {self.total_cost:,} EGP")
        print(f"  📈 Score Gain       : +{self.total_value} condition points")
        print(f"  🛣️  Roads Selected   : {len(self.selected_roads)}")
        print("─" * 60)
        print(f"  {'Road ID':<15} {'Condition':>10} {'Repair Cost':>12}")
        print(f"  {'─'*15} {'─'*10} {'─'*12}")
        for r in sorted(self.selected_roads, key=lambda x: x['condition']):
            bar = "🔴" if r['condition'] <= 4 else ("🟡" if r['condition'] <= 7 else "🟢")
            print(f"  {r['id']:<15} {bar} {r['condition']:>6}/10  {r['repair_cost']:>10,} EGP")
        print("═" * 60)

# ─────────────────────────────────────────────
# 3. DP KNAPSACK SOLVER
# ─────────────────────────────────────────────

def solve_maintenance(edges, budget):
    # نفلتر الطرق اللي محتاجة تتصلح وتكلفتها أقل من أو تساوي الميزانية
    candidates = [e for e in edges if e['repair_value'] > 0 and e['repair_cost'] <= budget]

    n = len(candidates)
    W = budget

    # DP Table: (n+1) x (W+1)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost  = candidates[i-1]['repair_cost']
        value = candidates[i-1]['repair_value']
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w] # احتمال 1: مانصلحش الطريق ده
            if w >= cost:
                # احتمال 2: نصلح الطريق لو الميزانية تكفي وناخد الأفضل
                dp[i][w] = max(dp[i][w], dp[i-1][w - cost] + value)

    # ── Backtracking لمعرفة الطرق اللي اخترناها ──
    selected = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(candidates[i-1])
            w -= candidates[i-1]['repair_cost']

    total_cost  = sum(r['repair_cost']  for r in selected)
    total_value = sum(r['repair_value'] for r in selected)

    return MaintenanceResult(budget, selected, total_cost, total_value, edges)

# ─────────────────────────────────────────────
# 4. GRAPHICAL CONDITION HEATMAP (X/Y)
# ─────────────────────────────────────────────

def plot_condition_heatmap(edges, nodes, result):
    print("\n🎨 Generating Graphical Heatmap...")
    G = nx.Graph()

    # Add Nodes
    pos = {}
    for node_id, data in nodes.items():
        G.add_node(node_id, label=data['name'])
        pos[node_id] = (data['x'], data['y'])

    # تقسيم الطرق حسب حالتها
    poor_edges, fair_edges, good_edges = [], [], []
    selected_edges = [(r['from_id'], r['to_id']) for r in result.selected_roads]

    for e in edges:
        edge_tuple = (e['from_id'], e['to_id'])
        G.add_edge(*edge_tuple)
        
        if e['condition'] <= 4:
            poor_edges.append(edge_tuple)
        elif e['condition'] <= 7:
            fair_edges.append(edge_tuple)
        else:
            good_edges.append(edge_tuple)

    plt.figure(figsize=(14, 10))

    # رسم النقط (Nodes)
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightgray', edgecolors='black')

    # رسم الطرق (Edges) حسب الحالة
    nx.draw_networkx_edges(G, pos, edgelist=poor_edges, width=2.0, edge_color='red', label='Poor Condition (1-4)')
    nx.draw_networkx_edges(G, pos, edgelist=fair_edges, width=2.0, edge_color='gold', label='Fair Condition (5-7)')
    nx.draw_networkx_edges(G, pos, edgelist=good_edges, width=2.0, edge_color='green', label='Good Condition (8-10)')

    # تمييز الطرق اللي الخوارزمية اختارت تصلحها
    nx.draw_networkx_edges(G, pos, edgelist=selected_edges, width=4.0, edge_color='blue', style='dashed', label='Selected for Repair (DP)')

    # إضافة أسماء الأماكن
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title(f"Cairo Smart City: Infrastructure Maintenance Heatmap\n(Budget: {result.budget:,} EGP)", fontsize=16)
    plt.legend(loc="best")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    def p(f): return os.path.join(BASE_DIR, f)

    try:
        edges = load_edges(p("edges.csv"))
        nodes = load_nodes(p("nodes.csv"))
        
        print(f"✅ Loaded {len(edges)} existing roads successfully.")

        # تحديد الميزانية (نقدر نغيرها عشان نشوف الخوارزمية هتتصرف إزاي)
        BUDGET = 2000 
        
        result = solve_maintenance(edges, BUDGET)
        result.summary()
        plot_condition_heatmap(edges, nodes, result)

    except FileNotFoundError:
        print("❌ Error: 'nodes.csv' or 'edges.csv' not found. Please ensure they are in the same directory as this script.")