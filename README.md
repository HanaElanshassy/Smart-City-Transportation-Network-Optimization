# Cairo Smart City Transportation Network Optimization

### CSE112 - Design and Analysis of Algorithms | Alamein International University

A smart transportation optimization system for Greater Cairo that combines graph algorithms, dynamic programming, greedy optimization, machine learning, Docker deployment, and an interactive Streamlit dashboard.

Live app:
[Smart City Transportation Network Optimization](https://smart-city-transportation-network-optimization.streamlit.app/)

Repository:
[HanaElanshassy/Smart-City-Transportation-Network-Optimization](https://github.com/HanaElanshassy/Smart-City-Transportation-Network-Optimization)

---

## Overview

This project models Cairo as a transportation graph where:

- Nodes represent neighborhoods, districts, and important facilities.
- Edges represent roads between locations.
- Road attributes include distance, capacity, condition, and construction cost.
- Traffic data is used to estimate congestion and dynamic travel time.

The system provides an interactive dashboard for:

- route planning,
- emergency response routing,
- road expansion planning,
- public transit allocation,
- road maintenance optimization,
- traffic signal simulation,
- network health monitoring.

---

## Main Features

| Feature | Algorithm / Technique | Module |
|---|---|---|
| Fastest route planning | Dijkstra's Algorithm | `dijkstra.py` |
| Emergency hospital routing | A* Search | `astar.py` |
| New road expansion | Kruskal's MST | `kruskal_mst.py` |
| Bus fleet allocation | Dynamic Programming | `public_transit_scheduler.py` |
| Road maintenance planning | 0/1 Knapsack DP | `dp_maintenance.py` |
| Traffic signal optimization | Greedy Algorithm | `greedy.py` |
| Traffic prediction | Random Forest ML | `traffic_ml_model.py` |
| Interactive dashboard | Streamlit + Plotly Mapbox | `app.py` |

---

## Streamlit Dashboard

The app contains four main pages:

### 1. Trip Planner

Finds optimal routes between Cairo locations using Dijkstra or A*.

### 2. Emergency Response

Compares Dijkstra and A* for emergency routing to hospitals and shows node-exploration efficiency.

### 3. Infrastructure

Includes:

- Road Expansion using Kruskal's MST.
- Bus Allocation using Dynamic Programming.
- Maintenance DP using 0/1 Knapsack.

### 4. System Dashboard

Shows congestion levels, network metrics, and traffic signal optimization results.

---

## Recent Improvements

The Maintenance DP tab was improved to match the rest of the project maps.

Before the update, the maintenance result was displayed as a separate graph-style heatmap. Now it uses the shared real Cairo map renderer used by the rest of the app.

Implemented improvements:

- Maintenance DP now appears on a real Plotly Mapbox Cairo basemap.
- Roads are colored by condition:
  - Poor: red/pink
  - Fair: yellow
  - Good: green
  - Selected for repair: cyan
- DP-selected repair roads are highlighted with thicker cyan lines and a glow effect.
- Hover details show condition, distance, repair cost, and score gain.
- Scroll zoom is enabled consistently across all real maps.
- Non-map charts were left unchanged.

Shared map config:

```python
PLOTLY_MAP_CONFIG = {"displayModeBar": False, "scrollZoom": True}
```

---

## Technical Reports

Two PDF reports are included in the repository:

| Report | Description |
|---|---|
| `Smart_City_Technical_Report.pdf` | General 5-page technical report |
| `Smart_City_Technical_Report_Code_Focused.pdf` | More code-focused report with implementation issues and fixes |

The code-focused report discusses issues such as:

- inconsistent maintenance map rendering,
- selected roads not being visually obvious,
- map zoom behavior,
- directed vs undirected road traversal,
- separating existing roads from potential roads,
- DP integer budget indexing,
- Streamlit rerun/session-state behavior,
- Docker runtime library requirements,
- dynamic cloud deployment ports.

---

## Project Structure

```text
Smart-City-Transportation-Network-Optimization/
├── app.py                                  # Streamlit dashboard
├── data_loader1.py                         # Core graph engine
├── dijkstra.py                             # Dijkstra shortest path
├── astar.py                                # A* emergency routing
├── kruskal_mst.py                          # Road expansion planner
├── dp_maintenance.py                       # Road maintenance DP knapsack
├── public_transit_scheduler.py             # Bus allocation DP
├── greedy.py                               # Traffic signal optimizer
├── traffic_ml_model.py                     # ML traffic predictor
├── traffic_api.py                          # TomTom API integration
├── simulation.py                           # Scenario testing
├── main.py                                 # Command-line integration runner
├── nodes.csv                               # Cairo locations
├── edges.csv                               # Road network
├── Traffic_Flow_Patterns.csv               # Traffic data
├── requirements.txt                        # Python dependencies
├── Dockerfile                              # Docker image definition
├── .dockerignore                           # Docker ignore rules
├── Smart_City_Technical_Report.pdf
└── Smart_City_Technical_Report_Code_Focused.pdf
```

---

## Run Locally

### 1. Clone The Repository

```bash
git clone https://github.com/HanaElanshassy/Smart-City-Transportation-Network-Optimization.git
cd Smart-City-Transportation-Network-Optimization
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run The Streamlit App

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Run With Docker

Make sure Docker Desktop is running.

### Build The Image

```bash
docker build -t smart-city-transportation .
```

### Run The Container

```bash
docker run --rm -p 8501:8501 smart-city-transportation
```

Open:

```text
http://localhost:8501
```

If port `8501` is busy:

```bash
docker run --rm -p 8502:8501 smart-city-transportation
```

Then open:

```text
http://localhost:8502
```

---

## Run Individual Algorithm Modules

```bash
python main.py
python main.py --module 3   # Dijkstra
python main.py --module 4   # A*
python main.py --module 5   # Transit DP
python main.py --module 6   # Maintenance DP
```

---

## Run Simulations

```bash
python simulation.py
python simulation.py --scenario 4   # Road closure and rerouting
python simulation.py --scenario 5   # Dijkstra vs A* comparison
python simulation.py --scenario 7   # Memoization speedup
```

Scenarios include:

1. Normal conditions
2. Morning rush hour
3. Evening rush hour
4. Road closure and rerouting
5. Emergency response comparison
6. Night free flow
7. Memoization performance test

---

## Live Traffic Data

The project supports optional TomTom traffic integration.

To use live traffic data:

1. Get an API key from [TomTom Developer Portal](https://developer.tomtom.com/).
2. Set the environment variable:

Windows:

```powershell
set TOMTOM_API_KEY=your_key_here
```

macOS/Linux:

```bash
export TOMTOM_API_KEY=your_key_here
```

3. Run:

```bash
python main.py --live
```

---

## Key Results

| Metric | Value |
|---|---|
| Graph nodes | 25 |
| Existing roads | 43 |
| Potential new roads | 12 |
| ML model R2 for congestion | Approximately 0.996 |
| A* efficiency | Explores fewer nodes than Dijkstra in emergency routing |
| Maintenance DP example | 12 roads selected, 1,973 / 2,000 cost-units used, +156 score gain |
| Docker support | Available |
| Streamlit deployment | Available |

---

## Requirements

See `requirements.txt` for exact versions.

Main dependencies:

- Python 3.11+
- Streamlit
- Plotly
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- networkx
- requests

---

## Team Roles

| Member | Role | Main Contribution |
|---|---|---|
| Member 1 | Graph Engine and ML | Data loading, Random Forest, traffic prediction |
| Member 2 | Infrastructure Designer | Kruskal's MST |
| Member 3 | Traffic Flow Engineer | Dijkstra routing |
| Member 4 | Emergency Response Lead | A* routing |
| Member 5 | Public Transit Scheduler | Bus allocation DP |
| Member 6 | Maintenance Engineer | DP Knapsack road repair |
| Member 7 | Signal Optimizer | Greedy signal control |

---

## License

This project was developed for educational purposes as part of the CSE112 Design and Analysis of Algorithms course.
