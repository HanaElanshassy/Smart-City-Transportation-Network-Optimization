"""
CSE112 - Member 1: Core Graph Engine
Cairo Smart City Transportation Network Optimization

Classes:
    Node       : a single location (neighborhood or facility)
    Edge       : a road connecting two nodes
    CairoGraph : the full weighted graph with ML-powered dynamic weights

Usage:
    from data_loader1 import CairoGraph
    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')
    graph.attach_predictor(predictor)   # from traffic_ml_model.py
"""

import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

class Node:
    def __init__(self, node_id, name, pop, node_type, x, y):
        self.id         = str(node_id)
        self.name       = str(name)
        self.type       = str(node_type)
        self.x          = float(x)
        self.y          = float(y)
        self.population = float(pop) if pd.notna(pop) else 0.0

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
# EDGE
# ─────────────────────────────────────────────

class Edge:
    def __init__(self, from_id, to_id, distance, capacity, condition, cost=0):
        self.from_id        = str(from_id)
        self.to_id          = str(to_id)
        self.distance       = float(distance)
        self.capacity       = float(capacity)
        self.condition      = float(condition)
        self.cost           = float(cost) if pd.notna(cost) else 0.0
        self.current_weight = self.distance   # updated by ML predictor

    @property
    def is_existing(self):
        return self.cost == 0 or self.from_id.startswith("F") or self.to_id.startswith("F")

    @property
    def is_potential(self):
        return not self.is_existing

    def __repr__(self):
        tag = "existing" if self.is_existing else f"potential(cost={self.cost}M)"
        return f"({self.from_id} <-> {self.to_id} | {self.distance}km | {tag})"


# ─────────────────────────────────────────────
# CAIRO GRAPH
# ─────────────────────────────────────────────

class CairoGraph:
    def __init__(self):
        self.nodes          = {}   # { node_id: Node }
        self.edges          = []   # all Edge objects
        self.adjacency_list = {}   # { node_id: [Edge, ...] }
        self._predictor     = None

    # ── Data Loading ─────────────────────────────────────────────

    def load_data(self, nodes_file='nodes.csv', edges_file='edges.csv'):
        nodes_df = pd.read_csv(nodes_file)
        for _, row in nodes_df.iterrows():
            node = Node(row['ID'], row['Name'], row['Population'], row['Type'], row['X'], row['Y'])
            self.nodes[node.id] = node
            self.adjacency_list[node.id] = []

        edges_df = pd.read_csv(edges_file)
        for _, row in edges_df.iterrows():
            from_id = str(row['FromID'])
            to_id   = str(row['ToID'])
            if from_id not in self.nodes or to_id not in self.nodes:
                continue

            cost     = row['Cost'] if 'Cost' in row.index else 0
            edge_fwd = Edge(from_id, to_id, row['Distance'], row['Capacity'], row['Condition'], cost)
            edge_rev = Edge(to_id, from_id, row['Distance'], row['Capacity'], row['Condition'], cost)

            self.edges.append(edge_fwd)
            self.edges.append(edge_rev)
            self.adjacency_list[from_id].append(edge_fwd)
            self.adjacency_list[to_id].append(edge_rev)

    # ── ML Predictor Integration ──────────────────────────────────

    def attach_predictor(self, predictor):
        """Attach ML traffic predictor. Once attached, get_weight() returns ML-driven values."""
        self._predictor = predictor

    def get_weight(self, from_id: str, to_id: str, time_of_day: str = "morning") -> float:
        """
        Returns dynamic edge weight (travel time in minutes).
        Uses ML predictor if attached, otherwise falls back to distance.
        Called by Members 3 (Dijkstra) and 4 (A*).
        """
        if self._predictor:
            return self._predictor.get_current_weight(from_id, to_id, time_of_day)
        edge = self.get_edge(from_id, to_id)
        return edge.distance if edge else float('inf')

    def get_congestion(self, from_id: str, to_id: str, time_of_day: str = "morning") -> float:
        """Returns congestion level [0.0 – 1.0]. Requires ML predictor."""
        if self._predictor:
            return self._predictor.get_congestion_level(from_id, to_id, time_of_day)
        return 0.5

    # ── Query Helpers ────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(str(node_id))

    def get_edges_from(self, node_id: str, existing_only: bool = True) -> list:
        edges = self.adjacency_list.get(str(node_id), [])
        return [e for e in edges if e.is_existing] if existing_only else edges

    def get_edge(self, from_id: str, to_id: str) -> Optional[Edge]:
        for edge in self.adjacency_list.get(str(from_id), []):
            if edge.to_id == str(to_id):
                return edge
        return None

    def get_existing_edges(self) -> list:
        seen, result = set(), []
        for edge in self.edges:
            if not edge.is_existing:
                continue
            key = tuple(sorted([edge.from_id, edge.to_id]))
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def get_potential_edges(self) -> list:
        seen, result = set(), []
        for edge in self.edges:
            if not edge.is_potential:
                continue
            key = tuple(sorted([edge.from_id, edge.to_id]))
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def get_nodes_by_type(self, node_type: str) -> list:
        return [n for n in self.nodes.values() if n.type == node_type]

    def get_high_population_nodes(self, top_n: int = 5) -> list:
        return sorted(self.nodes.values(), key=lambda n: n.population, reverse=True)[:top_n]

    def summary(self) -> dict:
        """Returns graph statistics as a dictionary."""
        return {
            "total_nodes":        len(self.nodes),
            "existing_roads":     len(self.get_existing_edges()),
            "potential_roads":    len(self.get_potential_edges()),
            "medical_facilities": [n.name for n in self.get_nodes_by_type("Medical")],
            "gov_centers":        [n.name for n in self.get_nodes_by_type("Government")],
            "ml_active":          self._predictor is not None,
        }
