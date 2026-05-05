"""
CSE112 - Member 1: TomTom Live Traffic API Integration
Cairo Smart City Transportation Network Optimization

Fetches real-time edge weights from TomTom's Routing + Traffic Flow APIs.

Setup:
    1. Get a free key at developer.tomtom.com
       Enable: Routing API + Traffic API (Traffic Flow)
    2. Set your key as an environment variable (NEVER paste it in code):
         Mac/Linux: export TOMTOM_API_KEY=your_key_here
         Windows  : set TOMTOM_API_KEY=your_key_here
         .env file: TOMTOM_API_KEY=your_key_here  ← add .env to .gitignore!
    3. Run:  python traffic_api.py

Public API:
    fetcher = TomTomFetcher()
    fetcher.fetch_all_edges(graph)
    df = fetcher.to_dataframe()
    fetcher.save_cache("live_weights.json")
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

TOMTOM_API_KEY   = os.environ.get("TOMTOM_API_KEY", "")
# ↑ Never paste your key here. Set it as an environment variable instead.

ROUTING_URL      = "https://api.tomtom.com/routing/1/calculateRoute"
TRAFFIC_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
REQUEST_DELAY    = 0.25


# ─────────────────────────────────────────────
# TOMTOM FETCHER
# ─────────────────────────────────────────────

class TomTomFetcher:
    """Fetches real-time travel times and congestion from TomTom for every edge in the Cairo graph."""

    TIME_SLOT_HOURS = {
        "morning":   8,
        "afternoon": 13,
        "evening":   18,
        "night":     23,
    }

    def __init__(self, api_key: str = TOMTOM_API_KEY):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "TomTom API key not set.\n"
                "Get a free key at: https://developer.tomtom.com\n"
                "Then set it as an environment variable:\n"
                "  Windows :  set TOMTOM_API_KEY=your_key_here\n"
                "  Mac/Linux: export TOMTOM_API_KEY=your_key_here\n"
                "Never paste the key directly into the source file."
            )
        self.api_key     = api_key
        self._cache      = {}
        self._call_count = 0

    # ── Core API Calls ───────────────────────────────────────────

    def _get_travel_time(self, from_x, from_y, to_x, to_y,
                          depart_at: Optional[str] = None) -> Optional[float]:
        coords = f"{from_y},{from_x}:{to_y},{to_x}"
        url    = f"{ROUTING_URL}/{coords}/json"
        params = {
            "key": self.api_key, "traffic": "true",
            "travelMode": "car", "routeType": "fastest",
        }
        if depart_at:
            params["departAt"] = depart_at
        try:
            resp = requests.get(url, params=params, timeout=10)
            self._call_count += 1
            if resp.status_code == 200:
                seconds = resp.json()["routes"][0]["summary"]["travelTimeInSeconds"]
                return round(seconds / 60, 2)
            if resp.status_code == 403:
                raise PermissionError("Invalid API key or Routing API not enabled.")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return None

    def _get_congestion(self, x, y) -> Optional[float]:
        params = {"key": self.api_key, "point": f"{y},{x}", "unit": "KMPH"}
        try:
            resp = requests.get(TRAFFIC_FLOW_URL, params=params, timeout=10)
            self._call_count += 1
            if resp.status_code == 200:
                data = resp.json()["flowSegmentData"]
                curr = data.get("currentSpeed", 0)
                free = data.get("freeFlowSpeed", 1)
                return round(max(0.0, min(1.0, 1.0 - curr / free)), 3) if free else 0.0
            if resp.status_code == 403:
                raise PermissionError("Invalid API key or Traffic API not enabled.")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return None

    def _depart_at_string(self, hour: int) -> str:
        return datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

    def _fallback_weight(self, distance_km: float, congestion: float = 0.5) -> float:
        speed_km_min = (60 - 50 * congestion) / 60
        return round(distance_km / speed_km_min, 2)

    # ── Main Fetch ───────────────────────────────────────────────

    def fetch_all_edges(self, graph, existing_only: bool = True) -> dict:
        """Fetch live travel times for every edge in the graph."""
        edges = graph.get_existing_edges() if existing_only else graph.edges

        for edge in edges:
            from_node = graph.get_node(edge.from_id)
            to_node   = graph.get_node(edge.to_id)
            if from_node is None or to_node is None:
                continue

            mid_x = (from_node.x + to_node.x) / 2
            mid_y = (from_node.y + to_node.y) / 2
            congestion_now = self._get_congestion(mid_x, mid_y)
            time.sleep(REQUEST_DELAY)

            for slot, hour in self.TIME_SLOT_HOURS.items():
                travel_time = self._get_travel_time(
                    from_node.x, from_node.y, to_node.x, to_node.y,
                    depart_at=self._depart_at_string(hour),
                )
                time.sleep(REQUEST_DELAY)

                cong = congestion_now if congestion_now is not None else 0.5
                if travel_time is None:
                    travel_time = self._fallback_weight(edge.distance, cong)

                if congestion_now is None:
                    base = edge.distance
                    cong = min(1.0, max(0.0, (travel_time - base) / (base * 5)))

                self._cache[f"{edge.from_id}->{edge.to_id}->{slot}"] = {
                    "from_id": edge.from_id, "to_id": edge.to_id,
                    "time_slot": slot, "travel_time": travel_time,
                    "congestion": round(cong, 3), "distance_km": edge.distance,
                }

        return self._cache

    # ── Public Weight API ────────────────────────────────────────

    def get_travel_time(self, from_id: str, to_id: str, time_slot: str = "morning") -> Optional[float]:
        result = (self._cache.get(f"{from_id}->{to_id}->{time_slot}") or
                  self._cache.get(f"{to_id}->{from_id}->{time_slot}"))
        return result["travel_time"] if result else None

    def get_congestion_level(self, from_id: str, to_id: str, time_slot: str = "morning") -> float:
        result = (self._cache.get(f"{from_id}->{to_id}->{time_slot}") or
                  self._cache.get(f"{to_id}->{from_id}->{time_slot}"))
        return result["congestion"] if result else 0.5

    # ── Export ───────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert cache to wide format matching Traffic_Flow_Patterns.csv."""
        rows = {}
        for key, data in self._cache.items():
            road_id = f"{data['from_id']}-{data['to_id']}"
            slot    = data["time_slot"].capitalize()
            if road_id not in rows:
                rows[road_id] = {"RoadID": road_id}
            rows[road_id][slot] = data["travel_time"]

        df = pd.DataFrame(list(rows.values()))
        for col in ["Morning", "Afternoon", "Evening", "Night"]:
            if col not in df.columns:
                df[col] = None
        return df[["RoadID", "Morning", "Afternoon", "Evening", "Night"]]

    def save_cache(self, filepath: str = "live_weights.json"):
        with open(filepath, "w") as f:
            json.dump(self._cache, f, indent=2)

    def load_cache(self, filepath: str = "live_weights.json") -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath) as f:
            self._cache = json.load(f)
        return True

    def save_as_traffic_csv(self, filepath: str = "Traffic_Flow_Patterns_live.csv"):
        """Save in Traffic_Flow_Patterns.csv format to replace static data for ML training."""
        self.to_dataframe().to_csv(filepath, index=False)


# ─────────────────────────────────────────────
# GRAPH INTEGRATION HELPER
# ─────────────────────────────────────────────

def attach_live_weights(graph, fetcher: TomTomFetcher, time_of_day: str = "morning"):
    """Update every edge's current_weight in the graph with live API data."""
    for edge in graph.edges:
        live_time = fetcher.get_travel_time(edge.from_id, edge.to_id, time_of_day)
        if live_time is not None:
            edge.current_weight = live_time