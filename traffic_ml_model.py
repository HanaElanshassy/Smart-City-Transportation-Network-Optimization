"""
CSE112 - Member 1: ML Traffic Prediction Model
Cairo Smart City Transportation Network Optimization

Trains a Random Forest model on Cairo's traffic flow data to predict:
    1. Congestion Level  (0.0 = free flow → 1.0 = fully congested)
    2. Dynamic Edge Weight (travel time in minutes, used by Dijkstra & A*)

Public API (used by Members 3 & 4):
    predictor = build_predictor()
    predictor.get_current_weight(from_id, to_id, time_of_day) -> float
    predictor.get_congestion_level(from_id, to_id, time_of_day) -> float

Note on model accuracy:
    The dataset contains ~112 samples (28 roads × 4 time slots). R² scores
    from cross-validation will appear very high because the model interpolates
    over a small, fixed dataset rather than generalising to unseen roads.
    This is sufficient for the project scope and is acknowledged in the report.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score

TIME_SLOTS = {
    "morning":   0,
    "afternoon": 1,
    "evening":   2,
    "night":     3,
}

MODEL_PATH = "traffic_model.pkl"

FEATURE_COLS = [
    "time_enc", "distance", "capacity", "condition",
    "flow", "flow_ratio", "from_type_enc", "to_type_enc",
    "from_pop", "to_pop", "is_peak",
]
TARGET_COLS = ["congestion", "dynamic_weight"]


# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare_data(
    traffic_file: str = "Traffic_Flow_Patterns.csv",
    edges_file:   str = "edges.csv",
    nodes_file:   str = "nodes.csv",
) -> tuple:
    traffic_df = pd.read_csv(traffic_file)
    edges_df   = pd.read_csv(edges_file)
    nodes_df   = pd.read_csv(nodes_file)

    edges_df["FromID"] = edges_df["FromID"].astype(str)
    edges_df["ToID"]   = edges_df["ToID"].astype(str)
    nodes_df["ID"]     = nodes_df["ID"].astype(str)

    # FIX: facility edges (F-prefixed nodes) are existing roads even if Cost > 0
    is_facility = edges_df["FromID"].str.startswith("F") | edges_df["ToID"].str.startswith("F")
    existing_edges = edges_df[(edges_df["Cost"] == 0) | is_facility].copy()

    all_types    = nodes_df["Type"].unique().tolist()
    type_encoder = {t: i for i, t in enumerate(sorted(all_types))}
    node_type    = dict(zip(nodes_df["ID"], nodes_df["Type"]))
    pop_map      = dict(zip(nodes_df["ID"], nodes_df["Population"].fillna(0)))

    records = []
    for _, row in traffic_df.iterrows():
        road_id = str(row["RoadID"])
        parts   = road_id.split("-")
        if len(parts) != 2:
            continue
        from_id, to_id = parts[0], parts[1]

        edge_row = existing_edges[
            (existing_edges["FromID"] == from_id) & (existing_edges["ToID"] == to_id)
        ]
        if edge_row.empty:
            edge_row = existing_edges[
                (existing_edges["FromID"] == to_id) & (existing_edges["ToID"] == from_id)
            ]
        if edge_row.empty:
            continue

        distance  = float(edge_row["Distance"].values[0])
        capacity  = float(edge_row["Capacity"].values[0])
        condition = float(edge_row["Condition"].values[0])

        from_type_enc = type_encoder.get(node_type.get(from_id, "Unknown"), -1)
        to_type_enc   = type_encoder.get(node_type.get(to_id,   "Unknown"), -1)
        from_pop      = pop_map.get(from_id, 0)
        to_pop        = pop_map.get(to_id,   0)

        for time_name, time_enc in TIME_SLOTS.items():
            flow       = float(row[time_name.capitalize()])
            congestion = min(flow / capacity, 1.0)

            avg_speed  = max(1.0 - 0.833 * congestion, 0.167)
            base_time  = distance / avg_speed
            cond_factor = 1.0 + (10 - condition) * 0.05
            dynamic_weight = base_time * cond_factor

            records.append({
                "from_id": from_id, "to_id": to_id,
                "road_id": road_id, "time_name": time_name,
                "time_enc": time_enc, "distance": distance,
                "capacity": capacity, "condition": condition,
                "flow": flow, "flow_ratio": flow / capacity,
                "from_type_enc": from_type_enc, "to_type_enc": to_type_enc,
                "from_pop": from_pop / 1e6, "to_pop": to_pop / 1e6,
                "is_peak": int(time_name in ("morning", "evening")),
                "congestion": congestion, "dynamic_weight": dynamic_weight,
            })

    return pd.DataFrame(records), type_encoder


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> MultiOutputRegressor:
    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=10,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
    )
    model.fit(X, y)
    return model


def evaluate_model(df: pd.DataFrame) -> dict:
    """Run 5-fold cross-validation and return R² scores per target."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values
    results = {}
    for i, target in enumerate(TARGET_COLS):
        rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X, y[:, i], cv=5, scoring="r2")
        results[target] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
    return results


# ─────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────

class TrafficPredictor:
    """
    Wrapper around the trained model.
    Exposes get_current_weight() and get_congestion_level()
    for use by Members 3 (Dijkstra) and 4 (A*).
    """

    def __init__(self, model, df, edges_df, nodes_df, type_encoder):
        self.model        = model
        self.type_encoder = type_encoder

        edges_df = edges_df.copy()
        edges_df["FromID"] = edges_df["FromID"].astype(str)
        edges_df["ToID"]   = edges_df["ToID"].astype(str)
        nodes_df = nodes_df.copy()
        nodes_df["ID"] = nodes_df["ID"].astype(str)

        # FIX: include facility edges (existing roads despite Cost > 0)
        is_facility  = edges_df["FromID"].str.startswith("F") | edges_df["ToID"].str.startswith("F")
        self._edges  = edges_df[(edges_df["Cost"] == 0) | is_facility]
        self._node_type = dict(zip(nodes_df["ID"], nodes_df["Type"]))
        self._node_pop  = dict(zip(nodes_df["ID"], nodes_df["Population"].fillna(0)))
        self._cache     = {}

        for _, row in df.iterrows():
            key = (row["from_id"], row["to_id"], row["time_name"])
            self._cache[key] = (row["congestion"], row["dynamic_weight"])

    def _build_feature_vector(self, from_id: str, to_id: str, time_slot: str) -> Optional[np.ndarray]:
        time_enc = TIME_SLOTS.get(time_slot.lower())
        if time_enc is None:
            raise ValueError(f"Unknown time_slot '{time_slot}'. Use: {list(TIME_SLOTS)}")

        edge = self._edges[
            (self._edges["FromID"] == from_id) & (self._edges["ToID"] == to_id)
        ]
        if edge.empty:
            edge = self._edges[
                (self._edges["FromID"] == to_id) & (self._edges["ToID"] == from_id)
            ]
        if edge.empty:
            return None

        distance  = float(edge["Distance"].values[0])
        capacity  = float(edge["Capacity"].values[0])
        condition = float(edge["Condition"].values[0])

        peak_ratio = {0: 0.90, 1: 0.50, 2: 0.85, 3: 0.25}
        flow       = capacity * peak_ratio[time_enc]
        flow_ratio = flow / capacity if capacity > 0 else 0

        from_type = self.type_encoder.get(self._node_type.get(from_id, "Unknown"), -1)
        to_type   = self.type_encoder.get(self._node_type.get(to_id,   "Unknown"), -1)
        from_pop  = self._node_pop.get(from_id, 0) / 1e6
        to_pop    = self._node_pop.get(to_id,   0) / 1e6
        is_peak   = int(time_slot.lower() in ("morning", "evening"))

        return np.array([[
            time_enc, distance, capacity, condition,
            flow, flow_ratio, from_type, to_type,
            from_pop, to_pop, is_peak,
        ]])

    def get_current_weight(self, from_id: str, to_id: str, time_of_day: str) -> float:
        """
        Returns dynamic edge weight (travel time in minutes).
        Used as the edge cost in Dijkstra (Member 3) and A* (Member 4).
        """
        key = (str(from_id), str(to_id), time_of_day.lower())
        if key in self._cache:
            return self._cache[key][1]

        features = self._build_feature_vector(*key)
        if features is None:
            edge = self._edges[
                (self._edges["FromID"] == key[0]) & (self._edges["ToID"] == key[1])
            ]
            if edge.empty:
                edge = self._edges[
                    (self._edges["FromID"] == key[1]) & (self._edges["ToID"] == key[0])
                ]
            dist = float(edge["Distance"].values[0]) if not edge.empty else 10.0
            return dist

        pred   = self.model.predict(features)[0]
        weight = float(pred[1])
        self._cache[key] = (float(pred[0]), weight)
        return weight

    def get_congestion_level(self, from_id: str, to_id: str, time_of_day: str) -> float:
        """Returns congestion level [0.0 – 1.0]."""
        key = (str(from_id), str(to_id), time_of_day.lower())
        if key in self._cache:
            return self._cache[key][0]

        features = self._build_feature_vector(*key)
        if features is None:
            return 0.5

        pred       = self.model.predict(features)[0]
        congestion = float(np.clip(pred[0], 0.0, 1.0))
        self._cache[key] = (congestion, float(pred[1]))
        return congestion

    def get_congestion_label(self, from_id: str, to_id: str, time_of_day: str) -> str:
        """Returns a human-readable congestion label."""
        c = self.get_congestion_level(from_id, to_id, time_of_day)
        if c < 0.4:   return "Low"
        if c < 0.7:   return "Moderate"
        if c < 0.9:   return "High"
        return "Severe"


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_model(model, type_encoder, path=MODEL_PATH):
    joblib.dump({"model": model, "type_encoder": type_encoder}, path)


def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data["model"], data["type_encoder"]


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def build_predictor(
    traffic_file:  str  = "Traffic_Flow_Patterns.csv",
    edges_file:    str  = "edges.csv",
    nodes_file:    str  = "nodes.csv",
    force_retrain: bool = False,
) -> TrafficPredictor:
    """
    One-liner setup for Members 3 & 4:
        from traffic_ml_model import build_predictor
        predictor = build_predictor()
        weight = predictor.get_current_weight("3", "F9", "morning")
    """
    edges_df = pd.read_csv(edges_file)
    nodes_df = pd.read_csv(nodes_file)

    if os.path.exists(MODEL_PATH) and not force_retrain:
        model, type_encoder = load_model()
        df, _ = load_and_prepare_data(traffic_file, edges_file, nodes_file)
    else:
        df, type_encoder = load_and_prepare_data(traffic_file, edges_file, nodes_file)
        model = train_model(df)
        save_model(model, type_encoder)

    return TrafficPredictor(model, df, edges_df, nodes_df, type_encoder)