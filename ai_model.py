"""
=============================================================
  CSE112 - Member 1: ML Traffic Prediction Model
  Cairo Smart City Transportation Network Optimization
=============================================================

This module trains a Random Forest model on Cairo's traffic
flow data to predict:
  1. Congestion Level  (0.0 = free flow → 1.0 = fully congested)
  2. Dynamic Edge Weight (travel time in minutes, used by Dijkstra & A*)

Public API (used by Members 3 & 4):
  get_current_weight(from_id, to_id, time_of_day) -> float
  get_congestion_level(from_id, to_id, time_of_day) -> float
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────
TIME_SLOTS = {
    "morning":   0,   # 07:00 – 10:00  (peak)
    "afternoon": 1,   # 10:00 – 17:00  (off-peak)
    "evening":   2,   # 17:00 – 20:00  (peak)
    "night":     3,   # 20:00 – 07:00  (free flow)
}

MODEL_PATH = "traffic_model.pkl"   # saved after training

# ─────────────────────────────────────────────
# 2.  DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare_data(
    traffic_file: str = "traffic.csv",
    edges_file:   str = "edges.csv",
    nodes_file:   str = "nodes.csv",
) -> pd.DataFrame:
    """
    Merge traffic patterns with road properties and node features
    to build a feature-rich training dataset.

    Returns a long-format DataFrame where each row = one road × one time slot.
    """

    # --- load raw files ---
    traffic_df = pd.read_csv(traffic_file)
    edges_df   = pd.read_csv(edges_file)
    nodes_df   = pd.read_csv(nodes_file)

    # Normalise ID types to string
    edges_df["FromID"] = edges_df["FromID"].astype(str)
    edges_df["ToID"]   = edges_df["ToID"].astype(str)
    nodes_df["ID"]     = nodes_df["ID"].astype(str)

    # Only keep EXISTING roads (Cost == 0 means existing, Cost > 0 means potential)
    existing_edges = edges_df[edges_df["Cost"] == 0].copy()

    # Build a node-type lookup (Residential / Business / Medical / …)
    node_type_map = dict(zip(nodes_df["ID"], nodes_df["Type"]))
    # Encode node types as integers for the model
    all_types = nodes_df["Type"].unique().tolist()
    type_encoder = {t: i for i, t in enumerate(sorted(all_types))}

    # Population lookup (facilities have 0)
    pop_map = dict(zip(nodes_df["ID"], nodes_df["Population"].fillna(0)))

    # --- melt traffic data from wide to long ---
    records = []
    for _, row in traffic_df.iterrows():
        road_id = str(row["RoadID"])
        parts   = road_id.split("-")
        if len(parts) != 2:
            continue
        from_id, to_id = parts[0], parts[1]

        # Match this road to its edge properties
        edge_row = existing_edges[
            (existing_edges["FromID"] == from_id) &
            (existing_edges["ToID"]   == to_id)
        ]
        if edge_row.empty:
            # try reverse direction (undirected graph)
            edge_row = existing_edges[
                (existing_edges["FromID"] == to_id) &
                (existing_edges["ToID"]   == from_id)
            ]
        if edge_row.empty:
            continue

        distance  = float(edge_row["Distance"].values[0])
        capacity  = float(edge_row["Capacity"].values[0])
        condition = float(edge_row["Condition"].values[0])

        from_type_enc = type_encoder.get(node_type_map.get(from_id, "Unknown"), -1)
        to_type_enc   = type_encoder.get(node_type_map.get(to_id,   "Unknown"), -1)
        from_pop      = pop_map.get(from_id, 0)
        to_pop        = pop_map.get(to_id,   0)

        for time_name, time_enc in TIME_SLOTS.items():
            flow = float(row[time_name.capitalize()])

            # ── Target 1: Congestion level (0 → 1)
            congestion = min(flow / capacity, 1.0)

            # ── Target 2: Dynamic weight = travel time in minutes
            # Base travel time (distance / avg speed in km/min)
            # Average speed degrades as congestion rises:
            #   free flow ≈ 60 km/h = 1 km/min
            #   fully congested ≈ 10 km/h = 0.167 km/min
            avg_speed_km_min = 1.0 - 0.833 * congestion  # linear degradation
            avg_speed_km_min = max(avg_speed_km_min, 0.167)
            base_travel_time = distance / avg_speed_km_min

            # Road condition penalty (poor condition → slower, more careful driving)
            condition_factor = 1.0 + (10 - condition) * 0.05   # up to +50% for cond=1

            dynamic_weight = base_travel_time * condition_factor

            records.append({
                # identifiers (not fed to model)
                "from_id":  from_id,
                "to_id":    to_id,
                "road_id":  road_id,
                "time_name": time_name,

                # ── FEATURES ──
                "time_enc":       time_enc,          # 0-3
                "distance":       distance,          # km
                "capacity":       capacity,          # veh/h
                "condition":      condition,         # 1-10
                "flow":           flow,              # veh/h at this time
                "flow_ratio":     flow / capacity,   # utilisation ratio
                "from_type_enc":  from_type_enc,
                "to_type_enc":    to_type_enc,
                "from_pop":       from_pop / 1e6,    # normalise to millions
                "to_pop":         to_pop   / 1e6,
                "is_peak":        int(time_name in ("morning", "evening")),

                # ── TARGETS ──
                "congestion":     congestion,
                "dynamic_weight": dynamic_weight,
            })

    df = pd.DataFrame(records)
    print(f"✅ Dataset ready: {len(df)} samples  ({len(df['road_id'].unique())} roads × 4 time slots)")
    return df, type_encoder


# ─────────────────────────────────────────────
# 3.  MODEL TRAINING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "time_enc", "distance", "capacity", "condition",
    "flow", "flow_ratio", "from_type_enc", "to_type_enc",
    "from_pop", "to_pop", "is_peak",
]
TARGET_COLS = ["congestion", "dynamic_weight"]


def train_model(df: pd.DataFrame) -> MultiOutputRegressor:
    """Train a multi-output Random Forest and evaluate with cross-validation."""

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    # Random Forest inside a MultiOutputRegressor wrapper
    base_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_rf)
    model.fit(X, y)

    # ── Cross-validation evaluation ──
    print("\n📊 Cross-Validation Results (5-fold):")
    for i, target in enumerate(TARGET_COLS):
        single_rf = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        scores = cross_val_score(single_rf, X, y[:, i], cv=5, scoring="r2")
        print(f"   {target:>15}  →  R² = {scores.mean():.4f}  (±{scores.std():.4f})")

    print("\n✅ Model trained successfully!")
    return model


# ─────────────────────────────────────────────
# 4.  PREDICTION ENGINE  (Public API)
# ─────────────────────────────────────────────

class TrafficPredictor:
    """
    Wrapper around the trained model.
    Exposes get_current_weight() and get_congestion_level()
    for use by Members 3 (Dijkstra) and 4 (A*).
    """

    def __init__(
        self,
        model: MultiOutputRegressor,
        df: pd.DataFrame,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        type_encoder: dict,
    ):
        self.model        = model
        self.type_encoder = type_encoder

        # Pre-build lookup tables from the DataFrames
        edges_df = edges_df.copy()
        edges_df["FromID"] = edges_df["FromID"].astype(str)
        edges_df["ToID"]   = edges_df["ToID"].astype(str)
        nodes_df = nodes_df.copy()
        nodes_df["ID"] = nodes_df["ID"].astype(str)

        self._edges    = edges_df[edges_df["Cost"] == 0]  # existing only
        self._node_type = dict(zip(nodes_df["ID"], nodes_df["Type"]))
        self._node_pop  = dict(zip(nodes_df["ID"], nodes_df["Population"].fillna(0)))

        # Cache: (from_id, to_id, time_slot) → [congestion, weight]
        self._cache: dict = {}

        # Pre-populate cache from training data
        for _, row in df.iterrows():
            key = (row["from_id"], row["to_id"], row["time_name"])
            self._cache[key] = (row["congestion"], row["dynamic_weight"])

    # ── internal helper ──────────────────────────────────────────
    def _build_feature_vector(
        self, from_id: str, to_id: str, time_slot: str
    ) -> np.ndarray | None:
        """Return feature array for a road+time pair, or None if unknown road."""

        time_enc = TIME_SLOTS.get(time_slot.lower())
        if time_enc is None:
            raise ValueError(f"Unknown time_slot '{time_slot}'. Use: {list(TIME_SLOTS)}")

        # Find edge (try both directions)
        edge = self._edges[
            (self._edges["FromID"] == from_id) & (self._edges["ToID"] == to_id)
        ]
        if edge.empty:
            edge = self._edges[
                (self._edges["FromID"] == to_id) & (self._edges["ToID"] == from_id)
            ]
        if edge.empty:
            return None  # road not in dataset

        distance  = float(edge["Distance"].values[0])
        capacity  = float(edge["Capacity"].values[0])
        condition = float(edge["Condition"].values[0])

        # Estimate flow via ratio of typical peak vs off-peak
        peak_ratio = {0: 0.90, 1: 0.50, 2: 0.85, 3: 0.25}
        flow = capacity * peak_ratio[time_enc]

        from_type = self.type_encoder.get(self._node_type.get(from_id, "Unknown"), -1)
        to_type   = self.type_encoder.get(self._node_type.get(to_id,   "Unknown"), -1)
        from_pop  = self._node_pop.get(from_id, 0) / 1e6
        to_pop    = self._node_pop.get(to_id,   0) / 1e6
        is_peak   = int(time_slot.lower() in ("morning", "evening"))

        flow_ratio = flow / capacity if capacity > 0 else 0

        return np.array([[
            time_enc, distance, capacity, condition,
            flow, flow_ratio, from_type, to_type,
            from_pop, to_pop, is_peak,
        ]])

    # ── Public API ───────────────────────────────────────────────

    def get_current_weight(self, from_id: str, to_id: str, time_of_day: str) -> float:
        """
        Returns the dynamic edge weight (travel time in minutes).
        Used as the edge cost in Dijkstra (Member 3) and A* (Member 4).

        Args:
            from_id     : Node ID string, e.g. "3" or "F9"
            to_id       : Node ID string
            time_of_day : "morning" | "afternoon" | "evening" | "night"

        Returns:
            float : estimated travel time in minutes
        """
        key = (str(from_id), str(to_id), time_of_day.lower())

        # 1. Check cache first (exact training data hit)
        if key in self._cache:
            return self._cache[key][1]

        # 2. Use model to predict
        features = self._build_feature_vector(*key)
        if features is None:
            # Unknown road → fall back to distance-only estimate (60 km/h)
            edge = self._edges[
                (self._edges["FromID"] == key[0]) & (self._edges["ToID"] == key[1])
            ]
            if edge.empty:
                edge = self._edges[
                    (self._edges["FromID"] == key[1]) & (self._edges["ToID"] == key[0])
                ]
            dist = float(edge["Distance"].values[0]) if not edge.empty else 10.0
            return dist / 1.0   # 1 km/min fallback

        pred = self.model.predict(features)[0]
        weight = float(pred[1])
        self._cache[key] = (float(pred[0]), weight)
        return weight

    def get_congestion_level(self, from_id: str, to_id: str, time_of_day: str) -> float:
        """
        Returns the congestion level as a ratio between 0.0 (free) and 1.0 (fully jammed).

        Args:
            from_id     : Node ID string
            to_id       : Node ID string
            time_of_day : "morning" | "afternoon" | "evening" | "night"

        Returns:
            float : congestion ratio [0.0 – 1.0]
        """
        key = (str(from_id), str(to_id), time_of_day.lower())

        if key in self._cache:
            return self._cache[key][0]

        features = self._build_feature_vector(*key)
        if features is None:
            return 0.5  # unknown road → assume moderate congestion

        pred = self.model.predict(features)[0]
        congestion = float(np.clip(pred[0], 0.0, 1.0))
        self._cache[key] = (congestion, float(pred[1]))
        return congestion

    def get_congestion_label(self, from_id: str, to_id: str, time_of_day: str) -> str:
        """Human-readable congestion label."""
        c = self.get_congestion_level(from_id, to_id, time_of_day)
        if c < 0.4:
            return "🟢 Low"
        elif c < 0.7:
            return "🟡 Moderate"
        elif c < 0.9:
            return "🔴 High"
        else:
            return "⛔ Severe"


# ─────────────────────────────────────────────
# 5.  SAVE / LOAD HELPERS
# ─────────────────────────────────────────────

def save_model(model, type_encoder, path=MODEL_PATH):
    joblib.dump({"model": model, "type_encoder": type_encoder}, path)
    print(f"💾 Model saved to '{path}'")


def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data["model"], data["type_encoder"]


# ─────────────────────────────────────────────
# 6.  FACTORY  (one-liner setup for other members)
# ─────────────────────────────────────────────

def build_predictor(
    traffic_file: str = "traffic.csv",
    edges_file:   str = "edges.csv",
    nodes_file:   str = "nodes.csv",
    force_retrain: bool = False,
) -> "TrafficPredictor":
    """
    Convenience factory used by Members 3 & 4:

        from traffic_ml_model import build_predictor
        predictor = build_predictor()
        weight = predictor.get_current_weight("3", "F9", "morning")

    Trains (or loads cached) model automatically.
    """
    edges_df = pd.read_csv(edges_file)
    nodes_df = pd.read_csv(nodes_file)

    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("📂 Loading cached model...")
        model, type_encoder = load_model()
        df, _ = load_and_prepare_data(traffic_file, edges_file, nodes_file)
    else:
        print("🏋️  Training new model...")
        df, type_encoder = load_and_prepare_data(traffic_file, edges_file, nodes_file)
        model = train_model(df)
        save_model(model, type_encoder)

    return TrafficPredictor(model, df, edges_df, nodes_df, type_encoder)


# ─────────────────────────────────────────────
# 7.  MAIN  (standalone test + demo)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── resolve file paths (works when run from any directory) ──
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    def p(f): return os.path.join(BASE_DIR, f)

    print("=" * 60)
    print("  CSE112 – ML Traffic Prediction Model")
    print("=" * 60)

    # Train / load
    predictor = build_predictor(
        traffic_file  = p("traffic.csv"),
        edges_file    = p("edges.csv"),
        nodes_file    = p("nodes.csv"),
        force_retrain = True,
    )

    # ── Demo: predictions for all time slots on key roads ──
    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)

    test_roads = [
        ("3",  "5",  "Downtown Cairo → Heliopolis"),
        ("1",  "3",  "Maadi → Downtown Cairo"),
        ("4",  "2",  "New Cairo → Nasr City"),
        ("7",  "8",  "6th Oct → Giza"),
        ("F1", "2",  "Airport → Nasr City"),
        ("F2", "3",  "Ramses Station → Downtown"),
        ("13", "4",  "New Admin Capital → New Cairo"),
    ]

    for from_id, to_id, label in test_roads:
        print(f"\n🛣️  {label}  ({from_id} → {to_id})")
        print(f"  {'Time':<12} {'Weight (min)':>14}  {'Congestion':>12}  Label")
        print(f"  {'-'*12} {'-'*14}  {'-'*12}  {'-'*15}")
        for slot in ["morning", "afternoon", "evening", "night"]:
            w = predictor.get_current_weight(from_id, to_id, slot)
            c = predictor.get_congestion_level(from_id, to_id, slot)
            lbl = predictor.get_congestion_label(from_id, to_id, slot)
            print(f"  {slot:<12} {w:>14.2f}  {c:>12.3f}  {lbl}")

    print("\n" + "=" * 60)
    print("  API USAGE EXAMPLE (for Members 3 & 4)")
    print("=" * 60)
    print("""
    from traffic_ml_model import build_predictor

    predictor = build_predictor()          # auto-trains or loads cache

    # In Dijkstra / A* – get edge cost:
    weight = predictor.get_current_weight("3", "9", "morning")

    # For dashboard / UI:
    level  = predictor.get_congestion_level("3", "9", "morning")
    label  = predictor.get_congestion_label("3", "9", "morning")
    """)
    print("✅ All done!")