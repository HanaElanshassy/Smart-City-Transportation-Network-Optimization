# 🚦 Cairo Smart City Transportation Network
### CSE112 — Design and Analysis of Algorithms | Alamein International University

A complete transportation optimization system for Greater Cairo implementing 7 algorithms across 7 team members, powered by real-time TomTom traffic data and machine learning.

---

## 🗺️ What This System Does

| Problem | Algorithm | Member |
|---|---|---|
| Which new roads should Cairo build? | Kruskal's MST | 2 |
| What's the fastest route right now? | Dijkstra's Algorithm | 3 |
| Fastest ambulance route to hospital? | A* Search | 4 |
| How to allocate 203 buses across 10 routes? | Dynamic Programming | 5 |
| Which roads to repair within budget? | DP Knapsack | 6 |
| How should traffic lights decide? | Greedy Algorithm | 7 |
| Predict travel times from traffic data? | Random Forest ML | 1 |

---

## 📁 Project Structure

```
cairo-transport/
├── data_loader1.py              # Core graph engine (Node, Edge, CairoGraph)
├── traffic_ml_model.py          # ML traffic predictor (Random Forest)
├── traffic_api.py               # TomTom live traffic API integration
├── dijkstra.py                  # Shortest path + memoized route cache
├── astar.py                     # Emergency routing + Dijkstra vs A* race
├── kruskal_mst.py               # Infrastructure expansion planner
├── dp_maintenance.py            # Road maintenance optimizer
├── public_transit_scheduler.py  # Bus allocation optimizer
├── greedy.py                    # Traffic signal controller
├── simulation.py                # Testing framework (7 scenarios)
├── main.py                      # Integration runner
├── add_hospital_edges.py        # Utility: add hospital road connections
├── nodes.csv                    # 25 locations (neighborhoods + facilities)
├── edges.csv                    # 55 roads (existing + potential new)
├── Traffic_Flow_Patterns.csv    # Static traffic data (4 time slots)
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Setup & Installation

### Option 1 — Run locally

```bash
# 1. Clone the repo
git clone https://github.com/your-team/cairo-transport.git
cd cairo-transport

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full demo
python main.py

# 4. Run a specific member's module
python main.py --module 4   # A* emergency routing
python main.py --module 7   # Greedy traffic signals
```

### Option 2 — Run with Docker (recommended)

```bash
# Build the image
docker build -t cairo-transport .

# Run the demo
docker run cairo-transport

# Run interactively
docker run -it cairo-transport bash
```

---

## 🌐 Live Traffic Data (TomTom API)

To use real Cairo traffic data instead of the static CSV:

```bash
# 1. Get a free key at developer.tomtom.com
#    Enable: Routing API + Traffic API (Traffic Flow)

# 2. Set your key as environment variable
set TOMTOM_API_KEY=your_key_here        # Windows
export TOMTOM_API_KEY=your_key_here     # Mac/Linux

# 3. Run with live data
python main.py --live
```

This fetches real travel times for all 43 existing roads × 4 time slots, trains the ML model on real data, and caches results to `live_weights.json`.

---

## 🧪 Running Simulations

```bash
python simulation.py                # all 7 scenarios
python simulation.py --scenario 4   # road closure test
python simulation.py --scenario 5   # Dijkstra vs A* race
python simulation.py --scenario 7   # memoization speedup test
```

**Scenarios:**
1. Normal conditions (afternoon)
2. Morning rush hour
3. Evening rush hour
4. Road closure + rerouting
5. Emergency response (Dijkstra vs A* comparison)
6. Night free flow
7. Memoization performance test

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Graph nodes | 25 (15 neighborhoods + 10 facilities) |
| Existing roads | 43 |
| Potential new roads | 12 |
| ML model R² (congestion) | 0.9962 |
| A* nodes explored vs Dijkstra | ~3–11 vs 25 (up to 22 fewer) |
| Memoization speedup | 60–107× on repeated queries |
| Bus fleet coverage | 3.4% (203 buses / 299K daily demand) |
| Maintenance budget used | 1,985 / 2,000 cost-units (99.25%) |

---

## 🏆 Bonus Features

- ✅ **ML-based traffic prediction** — Random Forest trained on temporal traffic data
- ✅ **Live TomTom API integration** — real Cairo congestion data replaces static CSV
- ✅ **Dijkstra vs A* race animation data** — `compare_algorithms()` exposes step-by-step exploration order for animation
- ✅ **Docker containerization** — runs identically on any machine
- ✅ **Memoized route cache** — 60–107× speedup on repeated route queries
- ✅ **Simulation framework** — 7 scenarios including road closure and emergency testing

---

## 👥 Team Members

| Member | Role | Algorithm |
|---|---|---|
| Member 1 | Graph Engine & ML | Data loading, Random Forest, TomTom API |
| Member 2 | Infrastructure Designer | Kruskal's MST |
| Member 3 | Traffic Flow Engineer | Dijkstra's Algorithm |
| Member 4 | Emergency Response Lead | A* Search |
| Member 5 | Public Transit Scheduler | Dynamic Programming (bus allocation) |
| Member 6 | Maintenance Engineer | DP Knapsack (road repair) |
| Member 7 | Signal Optimizer | Greedy Algorithm |

---

## 📋 Requirements

```
Python 3.11+
pandas, numpy, scikit-learn, joblib
requests (TomTom API)
streamlit (UI)
matplotlib, plotly, networkx
```

See `requirements.txt` for pinned versions.
