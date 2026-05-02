"""
=============================================================
  CSE112 - Member 1: Core Graph Engine
  Cairo Smart City Transportation Network Optimization
=============================================================

Provides:
  - Node       : a single location (neighborhood or facility)
  - Edge       : a road connecting two nodes
  - CairoGraph : the full graph with ML-powered dynamic weights

Usage by all members:
    from graph_file import CairoGraph
    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')
    graph.attach_predictor(predictor)   # from traffic_ml_model.py
"""

import pandas as pd


# ─────────────────────────────────────────────
# 1. NODE
# ─────────────────────────────────────────────

class Node:
    def __init__(self, node_id, name, pop, node_type, x, y):
        self.id   = str(node_id)
        self.name = str(name)
        self.type = str(node_type)
        self.x    = float(x)
        self.y    = float(y)

        # Facilities (hospitals, airports…) have no population → default to 0
        self.population = float(pop) if pd.notna(pop) else 0.0

    # Convenience flags used by MST and emergency routing
    @property
    def is_facility(self):
        return self.id.startswith("F")

    @property
    def is_medical(self):
        return self.type == "Medical"

    @property
    def is_government(self):
        return self.type == "Government"

    def __repr__(self):
        return f"[{self.id}] {self.name} ({self.type})"


# ─────────────────────────────────────────────
# 2. EDGE
# ─────────────────────────────────────────────

class Edge:
    def __init__(self, from_id, to_id, distance, capacity, condition, cost=0):
        self.from_id   = str(from_id)
        self.to_id     = str(to_id)
        self.distance  = float(distance)
        self.capacity  = float(capacity)
        self.condition = float(condition)

        # cost > 0  →  potential new road (construction cost in Million EGP)
        # cost == 0 →  existing road
        self.cost = float(cost) if pd.notna(cost) else 0.0

        # Dynamic weight (minutes). Starts as a distance-based estimate,
        # and gets updated by the ML predictor once attached.
        self.current_weight = self.distance   # 1 km/min fallback

    @property
    def is_existing(self):
        """True for current roads, False for potential new roads."""
        return self.cost == 0

    @property
    def is_potential(self):
        return self.cost > 0

    def __repr__(self):
        tag = "existing" if self.is_existing else f"potential(cost={self.cost}M)"
        return f"({self.from_id} <-> {self.to_id} | {self.distance}km | {tag})"


# ─────────────────────────────────────────────
# 3. CAIRO GRAPH
# ─────────────────────────────────────────────

class CairoGraph:
    def __init__(self):
        self.nodes          = {}   # { node_id: Node }
        self.edges          = []   # all Edge objects (forward only, undirected logic in adj)
        self.adjacency_list = {}   # { node_id: [Edge, ...] }  ← both directions stored here

        self._predictor     = None  # ML predictor (attached later)

    # ── Data Loading ────────────────────────────────────────────

    def load_data(self, nodes_file='nodes.csv', edges_file='edges.csv'):
        """
        Parse nodes.csv and edges.csv into the graph.
        edges.csv holds BOTH existing roads (Cost=0) and potential new roads (Cost>0).
        """
        print("Loading data from CSVs... 🚀")

        # 1. Load nodes
        nodes_df = pd.read_csv(nodes_file)
        for _, row in nodes_df.iterrows():
            node = Node(
                row['ID'], row['Name'],
                row['Population'], row['Type'],
                row['X'], row['Y']
            )
            self.nodes[node.id] = node
            self.adjacency_list[node.id] = []

        # 2. Load edges
        edges_df = pd.read_csv(edges_file)
        missing_nodes = set()

        for _, row in edges_df.iterrows():
            from_id = str(row['FromID'])
            to_id   = str(row['ToID'])

            # Validate both endpoints exist
            if from_id not in self.nodes:
                missing_nodes.add(from_id)
                continue
            if to_id not in self.nodes:
                missing_nodes.add(to_id)
                continue

            # FIX: use dict-style access with fallback for Cost column
            cost = row['Cost'] if 'Cost' in row.index else 0

            # Forward edge
            edge_fwd = Edge(from_id, to_id,
                            row['Distance'], row['Capacity'], row['Condition'], cost)
            self.edges.append(edge_fwd)
            self.adjacency_list[from_id].append(edge_fwd)

            # Reverse edge (undirected graph — both directions stored in adjacency list)
            edge_rev = Edge(to_id, from_id,
                            row['Distance'], row['Capacity'], row['Condition'], cost)
            self.edges.append(edge_rev)          # FIX: was missing before
            self.adjacency_list[to_id].append(edge_rev)

        if missing_nodes:
            print(f"⚠️  Warning: {len(missing_nodes)} node ID(s) in edges not found in nodes: {missing_nodes}")

        existing_count  = sum(1 for e in self.edges if e.is_existing)  // 2
        potential_count = sum(1 for e in self.edges if e.is_potential) // 2
        print(f"✅ Loaded {len(self.nodes)} nodes | "
              f"{existing_count} existing roads | "
              f"{potential_count} potential new roads")

    # ── ML Predictor Integration ─────────────────────────────────

    def attach_predictor(self, predictor):
        """
        Attach the ML traffic predictor (from traffic_ml_model.py).
        Once attached, get_weight() returns ML-driven dynamic weights
        instead of raw distance.

        Called by Member 1 after build_predictor():
            graph.attach_predictor(predictor)
        """
        self._predictor = predictor
        print("🤖 ML predictor attached. Dynamic weights are now active.")

    def get_weight(self, from_id: str, to_id: str, time_of_day: str = "morning") -> float:
        """
        Returns the dynamic edge weight (travel time in minutes).
        Uses ML predictor if attached, otherwise falls back to distance.

        This is the function Members 3 (Dijkstra) and 4 (A*) must call
        to get the cost of traversing an edge.

        Args:
            from_id     : e.g. "3" or "F9"
            to_id       : e.g. "5"
            time_of_day : "morning" | "afternoon" | "evening" | "night"

        Returns:
            float : travel time in minutes
        """
        if self._predictor:
            return self._predictor.get_current_weight(from_id, to_id, time_of_day)

        # Fallback: distance / 1 km per minute (= 60 km/h)
        edge = self.get_edge(from_id, to_id)
        return edge.distance if edge else float('inf')

    def get_congestion(self, from_id: str, to_id: str, time_of_day: str = "morning") -> float:
        """Returns congestion level [0.0 – 1.0]. Requires ML predictor."""
        if self._predictor:
            return self._predictor.get_congestion_level(from_id, to_id, time_of_day)
        return 0.5  # unknown without ML

    # ── Query Helpers (used by all members) ─────────────────────

    def get_node(self, node_id: str) -> Node | None:
        """Return Node by ID, or None if not found."""
        return self.nodes.get(str(node_id))

    def get_edges_from(self, node_id: str, existing_only: bool = True) -> list:
        """
        Return all edges leaving a node.

        Args:
            node_id       : source node ID
            existing_only : if True (default), skip potential new roads
        """
        edges = self.adjacency_list.get(str(node_id), [])
        if existing_only:
            return [e for e in edges if e.is_existing]
        return edges

    def get_edge(self, from_id: str, to_id: str) -> Edge | None:
        """Return the edge between two nodes (forward direction), or None."""
        for edge in self.adjacency_list.get(str(from_id), []):
            if edge.to_id == str(to_id):
                return edge
        return None

    def get_existing_edges(self) -> list:
        """Return all unique existing roads (one direction each)."""
        seen = set()
        result = []
        for edge in self.edges:
            if not edge.is_existing:
                continue
            key = tuple(sorted([edge.from_id, edge.to_id]))
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def get_potential_edges(self) -> list:
        """Return all unique potential new roads (one direction each)."""
        seen = set()
        result = []
        for edge in self.edges:
            if not edge.is_potential:
                continue
            key = tuple(sorted([edge.from_id, edge.to_id]))
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def get_nodes_by_type(self, node_type: str) -> list:
        """Return all nodes of a given type, e.g. 'Medical', 'Government'."""
        return [n for n in self.nodes.values() if n.type == node_type]

    def get_high_population_nodes(self, top_n: int = 5) -> list:
        """Return the top N nodes sorted by population (descending)."""
        return sorted(self.nodes.values(),
                      key=lambda n: n.population, reverse=True)[:top_n]

    # ── Graph Info ───────────────────────────────────────────────

    def summary(self):
        """Print a summary of the graph."""
        existing  = len(self.get_existing_edges())
        potential = len(self.get_potential_edges())
        medical   = self.get_nodes_by_type("Medical")
        gov       = self.get_nodes_by_type("Government")

        print("\n" + "="*50)
        print("  Cairo Transportation Graph — Summary")
        print("="*50)
        print(f"  Total nodes        : {len(self.nodes)}")
        print(f"  Existing roads     : {existing}")
        print(f"  Potential new roads: {potential}")
        print(f"  Medical facilities : {[n.name for n in medical]}")
        print(f"  Gov. centers       : {[n.name for n in gov]}")
        print(f"  ML predictor       : {'✅ Active' if self._predictor else '❌ Not attached'}")
        print("="*50)


# ─────────────────────────────────────────────
# 4. TESTING AREA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')
    graph.summary()

    # ── Test 1: Adjacency list for Downtown Cairo (ID: 3) ──
    print("\n🔍 Roads connected to Downtown Cairo (ID: 3):")
    for edge in graph.get_edges_from('3'):
        neighbor = graph.get_node(edge.to_id)
        print(f"   → {neighbor.name:<30} | {edge.distance} km | condition: {edge.condition}/10")

    # ── Test 2: High-population nodes ──
    print("\n👥 Top 5 highest-population areas:")
    for node in graph.get_high_population_nodes(5):
        print(f"   {node.name:<30} : {int(node.population):,}")

    # ── Test 3: Medical facilities ──
    print("\n🏥 Medical facilities:")
    for node in graph.get_nodes_by_type("Medical"):
        print(f"   {node}")

    # ── Test 4: ML integration (if predictor available) ──
    print("\n⚡ Testing get_weight() without ML predictor (distance fallback):")
    w = graph.get_weight("3", "5", "morning")
    print(f"   Downtown Cairo → Heliopolis (morning) = {w} km (fallback)")

    print("\n💡 To activate ML weights, do:")
    print("   from traffic_ml_model import build_predictor")
    print("   predictor = build_predictor()")
    print("   graph.attach_predictor(predictor)")
    print("   weight = graph.get_weight('3', '5', 'morning')  # now in minutes!")