"""
Cairo Smart City — Transportation Network Optimization
Streamlit Web Application  |  CSE112 Group Project

Run:
    streamlit run app.py
"""

import os
import sys
import math
import random
import copy

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Cairo SmartCity — Traffic AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset / Hide Streamlit chrome ───────── */
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }
[data-testid="stAppViewContainer"] {
    background: #060b14;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070d1a 0%, #0a1628 100%);
    border-right: 1px solid #0ff3;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060b14; }
::-webkit-scrollbar-thumb { background: #0ff6; border-radius: 2px; }

/* ── Typography ──────────────────────────── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
code, .mono     { font-family: 'JetBrains Mono', monospace !important; }

/* ── Sidebar label override ──────────────── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {
    color: #8ba8cc !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0d1e35 !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
    color: #e0f0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px #00d4ff22 !important;
}

/* ── Sidebar button ──────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #0066ff22) !important;
    border: 1px solid #00d4ff88 !important;
    color: #00d4ff !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #0066ff44) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 20px #00d4ff44 !important;
    transform: translateY(-1px) !important;
}

/* ── Chaos buttons ───────────────────────── */
.main-content .stButton > button {
    background: linear-gradient(135deg, #ff003322, #ff006622) !important;
    border: 1px solid #ff003388 !important;
    color: #ff4466 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.7rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.main-content .stButton > button:hover {
    background: linear-gradient(135deg, #ff003344, #ff006644) !important;
    box-shadow: 0 0 14px #ff003366 !important;
}

/* ── Reset chaos button ──────────────────── */
.reset-btn .stButton > button {
    background: linear-gradient(135deg, #00ff8822, #00cc6622) !important;
    border: 1px solid #00ff8888 !important;
    color: #00ff88 !important;
}
.reset-btn .stButton > button:hover {
    box-shadow: 0 0 14px #00ff8844 !important;
}

/* ── Dividers ────────────────────────────── */
hr { border-color: #0ff2 !important; margin: 1rem 0 !important; }

/* ── Tabs ────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0a1628;
    border-radius: 12px;
    border: 1px solid #1a3a5c;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    color: #4a7a9b !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #00d4ff22, #0066ff22) !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ── Toggle / checkbox ───────────────────── */
[data-testid="stCheckbox"] label {
    color: #8ba8cc !important;
    font-size: 0.82rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────

def page_header(title: str, subtitle: str, icon: str = ""):
    st.markdown(f"""
    <div style="
        padding: 0 0 1.8rem 0;
        border-bottom: 1px solid #0ff2;
        margin-bottom: 1.8rem;
    ">
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.7rem;
            color:#00d4ff88;
            letter-spacing:0.18em;
            text-transform:uppercase;
            margin-bottom:0.5rem;
        ">Cairo Smart City // CSE112</div>
        <h1 style="
            font-family:'Syne',sans-serif;
            font-size:2.1rem;
            font-weight:800;
            color:#e8f4ff;
            margin:0;
            letter-spacing:-0.01em;
            line-height:1.1;
        ">{icon} {title}</h1>
        <p style="
            color:#4a7a9b;
            font-size:0.88rem;
            margin:0.5rem 0 0 0;
            font-family:'JetBrains Mono',monospace;
        ">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", color: str = "#00d4ff", icon: str = ""):
    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #0d1e35, #0a1628);
        border: 1px solid {color}33;
        border-top: 2px solid {color};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position:absolute; top:-20px; right:-10px;
            font-size:3.5rem; opacity:0.06; pointer-events:none;
        ">{icon}</div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.65rem;
            color:{color}99;
            letter-spacing:0.14em;
            text-transform:uppercase;
            margin-bottom:0.5rem;
        ">{label}</div>
        <div style="
            font-family:'Syne',sans-serif;
            font-size:1.85rem;
            font-weight:800;
            color:{color};
            line-height:1;
            margin-bottom:0.3rem;
        ">{value}</div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.7rem;
            color:#4a7a9b;
        ">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, accent: str = "#00d4ff"):
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.7rem;
        margin: 1.5rem 0 0.8rem 0;
    ">
        <div style="
            width:3px; height:18px;
            background:{accent};
            border-radius:2px;
            box-shadow: 0 0 8px {accent};
        "></div>
        <span style="
            font-family:'Syne',sans-serif;
            font-size:0.95rem;
            font-weight:700;
            color:#c8e0f4;
            letter-spacing:0.04em;
            text-transform:uppercase;
        ">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def status_badge(label: str, color: str):
    st.markdown(f"""
    <span style="
        display:inline-block;
        background:{color}22;
        border:1px solid {color}66;
        color:{color};
        border-radius:20px;
        padding:0.2rem 0.7rem;
        font-size:0.7rem;
        font-family:'JetBrains Mono',monospace;
        letter-spacing:0.06em;
    ">{label}</span>
    """, unsafe_allow_html=True)


def sidebar_section(title: str):
    st.sidebar.markdown(f"""
    <div style="
        font-family:'JetBrains Mono',monospace;
        font-size:0.6rem;
        color:#00d4ff55;
        letter-spacing:0.22em;
        text-transform:uppercase;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom:0.4rem;
        border-bottom:1px solid #0ff1;
    ">{title}</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING  (cached — runs once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_graph():
    from data_loader1 import CairoGraph
    graph = CairoGraph()
    graph.load_data(
        os.path.join(BASE_DIR, "nodes.csv"),
        os.path.join(BASE_DIR, "edges.csv"),
    )
    try:
        from traffic_ml_model import build_predictor
        traffic_csv = (
            os.path.join(BASE_DIR, "Traffic_Flow_Patterns_live.csv")
            if os.path.exists(os.path.join(BASE_DIR, "Traffic_Flow_Patterns_live.csv"))
            else os.path.join(BASE_DIR, "Traffic_Flow_Patterns.csv")
        )
        predictor = build_predictor(
            traffic_csv,
            os.path.join(BASE_DIR, "edges.csv"),
            os.path.join(BASE_DIR, "nodes.csv"),
        )
        graph.attach_predictor(predictor)
    except Exception:
        pass
    return graph


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "route_result":      None,
        "chaos_edges":       {},      # { "from_id->to_id": float congestion override }
        "chaos_log":         [],
        "active_page":       "Trip Planner",
        "last_src":          None,
        "last_tgt":          None,
        "last_time":         "morning",
        "show_potential":    False,
        "algo_choice":       "Dijkstra",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# MAP BUILDER
# ─────────────────────────────────────────────

CONGESTION_PALETTE = {
    "low":      "#00ff88",   # green
    "moderate": "#ffcc00",   # amber
    "high":     "#ff6600",   # orange
    "severe":   "#ff0033",   # red
    "path":     "#00d4ff",   # neon blue — optimal route
    "chaos":    "#ff0080",   # magenta — chaos-triggered edge
}

NODE_PALETTE = {
    "Medical":    "#ff3366",
    "Government": "#aa44ff",
    "Business":   "#00aaff",
    "Mixed":      "#00ccaa",
    "Residential":"#4488cc",
    "Industrial": "#ff8800",
    "Airport":    "#ffdd00",
    "Transit Hub":"#ff8844",
    "Education":  "#44ddff",
    "Tourism":    "#ff44aa",
    "Sports":     "#44ff88",
    "Commercial": "#ffaa00",
}

def _congestion_color(level: float) -> str:
    if level < 0.40: return CONGESTION_PALETTE["low"]
    if level < 0.65: return CONGESTION_PALETTE["moderate"]
    if level < 0.85: return CONGESTION_PALETTE["high"]
    return CONGESTION_PALETTE["severe"]


def _congestion_label(level: float) -> str:
    if level < 0.40: return "LOW"
    if level < 0.65: return "MODERATE"
    if level < 0.85: return "HIGH"
    return "SEVERE"


def build_map(
    graph,
    time_of_day:   str  = "morning",
    route_path:    list = None,
    chaos_edges:   dict = None,
    show_potential:bool = False,
) -> go.Figure:
    """
    Build the Plotly interactive Cairo map.
    - All edges coloured by ML congestion level
    - Chaos edges coloured magenta
    - Optimal route drawn as thick glowing neon line
    - Nodes plotted with type-based colour + popup info
    """
    chaos_edges = chaos_edges or {}
    route_path  = route_path  or []
    route_set   = set(zip(route_path, route_path[1:])) if len(route_path) > 1 else set()

    fig = go.Figure()

    # ── 1. Draw edges ──────────────────────────────────────────
    edges = graph.get_existing_edges()
    if show_potential:
        edges = edges + graph.get_potential_edges()

    for edge in edges:
        n_from = graph.get_node(edge.from_id)
        n_to   = graph.get_node(edge.to_id)
        if not n_from or not n_to:
            continue

        chaos_key  = f"{edge.from_id}->{edge.to_id}"
        chaos_key2 = f"{edge.to_id}->{edge.from_id}"
        is_chaos   = chaos_key in chaos_edges or chaos_key2 in chaos_edges
        is_route   = (
            (edge.from_id, edge.to_id) in route_set or
            (edge.to_id, edge.from_id) in route_set
        )
        is_potential = edge.is_potential

        if is_chaos:
            congestion = chaos_edges.get(chaos_key, chaos_edges.get(chaos_key2, 0.95))
        elif is_potential:
            congestion = 0.0
        else:
            congestion = graph.get_congestion(edge.from_id, edge.to_id, time_of_day)

        color = (
            CONGESTION_PALETTE["chaos"] if is_chaos else
            ("#2244aa"                  if is_potential else
             _congestion_color(congestion))
        )

        lx = [n_from.x, n_to.x, None]
        ly = [n_from.y, n_to.y, None]
        w  = 1.5 if is_potential else 2.5

        hover = (
            f"{n_from.name} → {n_to.name}<br>"
            f"Distance: {edge.distance} km<br>"
            f"Condition: {edge.condition}/10<br>"
            f"Congestion: {_congestion_label(congestion)}"
            + (" ⚠️ CHAOS" if is_chaos else "")
            + (" [potential]" if is_potential else "")
        )

        fig.add_trace(go.Scattermap(
            lon=lx, lat=ly, mode="lines",
            line=dict(color=color, width=w),
            hoverinfo="text", hovertext=hover,
            showlegend=False,
        ))

    # ── 2. Draw route glow (thick + thin layered) ──────────────
    if len(route_path) > 1:
        rx = []
        ry = []
        for nid in route_path:
            n = graph.get_node(nid)
            if n:
                rx.append(n.x)
                ry.append(n.y)

        # Outer glow
        fig.add_trace(go.Scattermap(
            lon=rx, lat=ry, mode="lines",
            line=dict(color="#00d4ff", width=12),
            opacity=0.18,
            hoverinfo="skip", showlegend=False,
        ))
        # Mid glow
        fig.add_trace(go.Scattermap(
            lon=rx, lat=ry, mode="lines",
            line=dict(color="#00d4ff", width=6),
            opacity=0.45,
            hoverinfo="skip", showlegend=False,
        ))
        # Core line
        fig.add_trace(go.Scattermap(
            lon=rx, lat=ry, mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            opacity=1.0,
            name="Optimal Route",
            hoverinfo="skip",
        ))

    # ── 3. Draw nodes ──────────────────────────────────────────
    by_type: dict = {}
    for nid, node in graph.nodes.items():
        t = node.type
        if t not in by_type:
            by_type[t] = {"lons":[], "lats":[], "texts":[], "ids":[]}
        congestion_str = ""
        if not node.is_facility:
            pass
        pop_str = f"Pop: {int(node.population):,}" if node.population > 0 else ""
        by_type[t]["lons"].append(node.x)
        by_type[t]["lats"].append(node.y)
        by_type[t]["texts"].append(
            f"<b>{node.name}</b><br>"
            f"ID: {node.id}<br>"
            f"Type: {node.type}<br>"
            + (f"{pop_str}" if pop_str else "")
        )
        by_type[t]["ids"].append(nid)

    for t, data in by_type.items():
        color = NODE_PALETTE.get(t, "#aaccee")
        is_route_node = any(nid in route_path for nid in data["ids"])
        fig.add_trace(go.Scattermap(
            lon=data["lons"], lat=data["lats"],
            mode="markers+text",
            marker=dict(
                size=10 if t in ("Medical",) else 8,
                color=color,
                opacity=0.95,
                symbol="circle",
            ),
            text=[g.split("<br>")[0].replace("<b>","").replace("</b>","")
                  for g in data["texts"]],
            textposition="top right",
            textfont=dict(color="#c8e0f4", size=8),
            hovertext=data["texts"],
            hoverinfo="text",
            name=t,
        ))

    # ── 4. Highlight route waypoints ───────────────────────────
    if route_path:
        rn = [graph.get_node(nid) for nid in route_path if graph.get_node(nid)]
        fig.add_trace(go.Scattermap(
            lon=[n.x for n in rn],
            lat=[n.y for n in rn],
            mode="markers",
            marker=dict(size=14, color="#00d4ff", symbol="circle",
                        opacity=0.9),
            hovertext=[f"<b>{n.name}</b>" for n in rn],
            hoverinfo="text",
            name="Route Stops",
            showlegend=True,
        ))
        # Start (green) and End (red) markers
        if len(rn) >= 2:
            fig.add_trace(go.Scattermap(
                lon=[rn[0].x], lat=[rn[0].y], mode="markers",
                marker=dict(size=18, color="#00ff88", symbol="circle"),
                hovertext=[f"<b>START</b>: {rn[0].name}"],
                hoverinfo="text", name="Start", showlegend=False,
            ))
            fig.add_trace(go.Scattermap(
                lon=[rn[-1].x], lat=[rn[-1].y], mode="markers",
                marker=dict(size=18, color="#ff3366", symbol="circle"),
                hovertext=[f"<b>END</b>: {rn[-1].name}"],
                hoverinfo="text", name="Destination", showlegend=False,
            ))

    # ── 5. Layout ──────────────────────────────────────────────
    centre_lat = 30.03
    centre_lon = 31.25

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=centre_lat, lon=centre_lon),
            zoom=11.2,
        ),
        paper_bgcolor="#060b14",
        plot_bgcolor="#060b14",
        margin=dict(l=0, r=0, t=0, b=0),
        height=580,
        legend=dict(
            bgcolor='rgba(10, 22, 40, 0.8)',
            bordercolor="#1a3a5c",
            borderwidth=1,
            font=dict(color="#8ba8cc", size=10, family="JetBrains Mono"),
            x=0.01, y=0.99,
        ),
        hoverlabel=dict(
            bgcolor="#0d1e35",
            bordercolor="#1a3a5c",
            font=dict(color="#e0f0ff", family="JetBrains Mono", size=11),
        ),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(graph):
    # Logo / branding
    st.sidebar.markdown("""
    <div style="
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid #0ff2;
        margin-bottom: 0.5rem;
    ">
        <div style="
            font-family:'Syne',sans-serif;
            font-size:1.15rem;
            font-weight:800;
            color:#e8f4ff;
            letter-spacing:-0.01em;
        ">🏙️ CairoNet AI</div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.62rem;
            color:#00d4ff66;
            letter-spacing:0.15em;
            text-transform:uppercase;
            margin-top:0.2rem;
        ">Smart City Transport v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Page nav
    sidebar_section("Navigation")
    pages = ["Trip Planner", "Emergency Response", "Infrastructure", "System Dashboard"]
    page_icons = ["🗺️", "🚑", "🏗️", "📊"]
    for icon, pg in zip(page_icons, pages):
        active = st.session_state.active_page == pg
        if st.sidebar.button(
            f"{icon}  {pg}",
            key=f"nav_{pg}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.active_page = pg
            st.rerun()

    st.sidebar.divider()

    # Route controls
    node_items = sorted(
        [(nid, graph.get_node(nid).name) for nid in graph.nodes],
        key=lambda x: x[1]
    )
    node_labels = [f"{name}  [{nid}]" for nid, name in node_items]
    node_ids    = [nid for nid, _ in node_items]

    # Smart defaults
    try:
        default_src = node_ids.index("4")   # New Cairo
        default_tgt = node_ids.index("F9")  # Qasr El Aini
    except ValueError:
        default_src, default_tgt = 0, 1

    src_id, tgt_id, time_sel, algo_sel, find_clicked = None, None, "morning", "Dijkstra", False

    if st.session_state.active_page == "Trip Planner":
        sidebar_section("Route Planning")
        src_label = st.sidebar.selectbox("Origin Node",    node_labels, index=default_src, key="src_sel")
        tgt_label = st.sidebar.selectbox("Destination",    node_labels, index=default_tgt, key="tgt_sel")
        time_sel  = st.sidebar.selectbox("Time of Day",    ["morning","afternoon","evening","night"], key="time_sel")
        algo_sel  = st.sidebar.selectbox("Algorithm",      ["Dijkstra","A* (Emergency)"], key="algo_sel")

        src_id = node_ids[node_labels.index(src_label)]
        tgt_id = node_ids[node_labels.index(tgt_label)]

        find_clicked = st.sidebar.button("⚡  Find Optimal Route", use_container_width=True)

        st.sidebar.divider()

        sidebar_section("Map Layers")
        show_potential = st.sidebar.checkbox("Show Potential New Roads", value=False, key="show_pot")
        st.session_state.show_potential = show_potential

    # System status
    st.sidebar.divider()
    sidebar_section("System Status")
    ml_active = graph._predictor is not None
    status_color = "#00ff88" if ml_active else "#ff6600"
    status_text  = "ML ONLINE" if ml_active else "FALLBACK"
    st.sidebar.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.5rem;
        font-family:'JetBrains Mono',monospace;
        font-size:0.72rem; color:{status_color};
    ">
        <div style="
            width:7px; height:7px; border-radius:50%;
            background:{status_color};
            box-shadow:0 0 6px {status_color};
            animation: pulse 2s infinite;
        "></div>
        Traffic AI · {status_text}
    </div>
    <div style="
        font-family:'JetBrains Mono',monospace;
        font-size:0.68rem; color:#2a5a7a;
        margin-top:0.3rem;
    ">
        Nodes: {len(graph.nodes)} · Roads: {len(graph.get_existing_edges())}
    </div>
    <style>
    @keyframes pulse {{
        0%,100% {{ opacity:1; }}
        50%      {{ opacity:0.3; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    return src_id, tgt_id, time_sel, algo_sel, find_clicked


# ─────────────────────────────────────────────
# CHAOS SIMULATOR
# ─────────────────────────────────────────────

CHAOS_EVENTS = {
    "🔥 Ring Road Accident":    [("7","8"), ("8","7")],
    "🌧️  Flash Flood – Giza":   [("8","10"),("8","12"),("1","8")],
    "🚧 Downtown Closure":      [("3","2"), ("2","3"), ("3","5")],
    "⚡ Signal Failure Hub":    [("3","9"), ("3","6"), ("3","10")],
    "🏟️  Stadium Event":        [("2","5"), ("5","11"), ("3","5")],
    "🚨 Airport Emergency":     [("F1","5"),("F1","2")],
}


def render_chaos_section(graph):
    section_header("Chaos Simulator", "#ff3366")
    st.markdown("""
    <p style='
        font-family:"JetBrains Mono",monospace;
        font-size:0.72rem; color:#4a5a6a;
        margin-bottom:0.8rem;
    '>Inject real-world crises — congestion maxes out instantly and routes recalculate.</p>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (event_name, affected_edges) in enumerate(CHAOS_EVENTS.items()):
        with cols[i % 3]:
            if st.button(event_name, key=f"chaos_{i}", use_container_width=True):
                for from_id, to_id in affected_edges:
                    key = f"{from_id}->{to_id}"
                    st.session_state.chaos_edges[key] = random.uniform(0.88, 1.0)
                st.session_state.chaos_log.append(f"⚠️ {event_name} triggered")
                st.rerun()

    # Reset
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("✅  Reset All Chaos", key="chaos_reset", use_container_width=False):
        st.session_state.chaos_edges = {}
        st.session_state.chaos_log.append("✅ All chaos cleared")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Log
    if st.session_state.chaos_log:
        with st.expander("Event Log", expanded=False):
            for entry in reversed(st.session_state.chaos_log[-8:]):
                color = "#ff4466" if "⚠️" in entry else "#00ff88"
                st.markdown(f"""
                <div style='
                    font-family:"JetBrains Mono",monospace;
                    font-size:0.72rem; color:{color};
                    padding:0.2rem 0;
                '>{entry}</div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: TRIP PLANNER
# ─────────────────────────────────────────────

def page_trip_planner(graph, src_id, tgt_id, time_sel, algo_sel, find_clicked):
    page_header(
        "Trip Planner",
        "AI-powered routing with real-time congestion weights",
        "🗺️",
    )

    # Trigger route calculation
    if find_clicked and src_id != tgt_id:
        with st.spinner(""):
            if algo_sel == "A* (Emergency)":
                from astar import get_emergency_path
                result = get_emergency_path(graph, src_id, tgt_id, time_sel)
            else:
                from dijkstra import get_shortest_path
                result = get_shortest_path(graph, src_id, tgt_id, time_sel)
            st.session_state.route_result = result
            st.session_state.last_src  = src_id
            st.session_state.last_tgt  = tgt_id
            st.session_state.last_time = time_sel

    result = st.session_state.route_result
    route_path = result.path if result and result.found else []

    # ── KPI metrics ─────────────────────────────────────────────
    if result and result.found:
        # compute avg congestion along path
        cong_vals = []
        for a, b in zip(route_path, route_path[1:]):
            c = graph.get_congestion(a, b, st.session_state.last_time)
            cong_vals.append(c)
        avg_cong = sum(cong_vals) / len(cong_vals) if cong_vals else 0

        # total distance
        total_dist = sum(
            (graph.get_edge(a, b) or graph.get_edge(b, a)).distance
            for a, b in zip(route_path, route_path[1:])
            if graph.get_edge(a, b) or graph.get_edge(b, a)
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Travel Time", f"{result.total_time:.1f}m",
                     "AI-weighted estimate", "#00d4ff", "⏱️")
        with c2:
            kpi_card("Total Distance", f"{total_dist:.1f} km",
                     "Along optimal path", "#00ff88", "📏")
        with c3:
            cong_pct = avg_cong * 100
            cong_color = "#ff3366" if cong_pct > 70 else ("#ffcc00" if cong_pct > 40 else "#00ff88")
            kpi_card("Avg Congestion", f"{cong_pct:.0f}%",
                     _congestion_label(avg_cong).title(), cong_color, "🚦")
        with c4:
            kpi_card("Nodes Explored", f"{result.nodes_explored}",
                     f"{algo_sel} algorithm", "#aa44ff", "🔍")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Travel Time",    "—", "Select route", "#00d4ff44", "⏱️")
        with c2: kpi_card("Total Distance", "—", "Select route", "#00ff8844", "📏")
        with c3: kpi_card("Avg Congestion", "—", "Select route", "#ffcc0044", "🚦")
        with c4: kpi_card("Nodes Explored", "—", "Select route", "#aa44ff44", "🔍")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Map + Chaos side by side ─────────────────────────────────
    map_col, detail_col = st.columns([3, 1])

    with map_col:
        fig = build_map(
            graph,
            time_of_day=st.session_state.last_time,
            route_path=route_path,
            chaos_edges=st.session_state.chaos_edges,
            show_potential=st.session_state.show_potential,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with detail_col:
        section_header("Route Details", "#00d4ff")
        if result and result.found:
            path_names = result.path_names(graph)
            for i, name in enumerate(path_names):
                is_start = i == 0
                is_end   = i == len(path_names) - 1
                dot_color  = "#00ff88" if is_start else ("#ff3366" if is_end else "#00d4ff")
                label_color = "#e0f0ff"
                connector = "" if is_end else (
                    "<div style='margin-left:0.45rem; width:1px; height:14px;"
                    f"background:linear-gradient(#00d4ff44,#00d4ff11)'></div>"
                )
                html_string = f"""<div style='display:flex; align-items:flex-start; gap:0.5rem; margin-bottom:0;'>
<div style='display:flex;flex-direction:column;align-items:center;'>
<div style='width:10px; height:10px; border-radius:50%; background:{dot_color}; box-shadow:0 0 6px {dot_color}; margin-top:2px; flex-shrink:0;'></div>
{connector}
</div>
<div style='font-family:"JetBrains Mono",monospace; font-size:0.73rem; color:{label_color}; padding-bottom:0.6rem;'>{name}</div>
</div>"""
                st.markdown(html_string, unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='
                background:#0d1e35; border:1px solid #1a3a5c;
                border-radius:8px; padding:0.8rem;
                font-family:"JetBrains Mono",monospace; font-size:0.7rem; color:#4a7a9b;
            '>
                {len(path_names)} stops · {algo_sel}<br>
                Time: {st.session_state.last_time.capitalize()}<br>
                Chaos active: {len(st.session_state.chaos_edges)} edge(s)
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='
                background:#0d1e3544; border:1px solid #1a3a5c44;
                border-radius:8px; padding:1rem;
                font-family:"JetBrains Mono",monospace; font-size:0.75rem;
                color:#2a4a6a; text-align:center; margin-top:0.5rem;
            '>
                Select origin &amp; destination,<br>then click<br>
                <span style='color:#00d4ff66'>⚡ Find Optimal Route</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Chaos Simulator ─────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    render_chaos_section(graph)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: EMERGENCY RESPONSE
# ─────────────────────────────────────────────

def page_emergency(graph):
    page_header("Emergency Response", "Dijkstra vs A* — node exploration race", "🚑")

    from dijkstra import get_shortest_path
    from astar    import get_emergency_path

    medical = graph.get_nodes_by_type("Medical")
    all_nodes = sorted(
        [(nid, graph.get_node(nid).name) for nid in graph.nodes],
        key=lambda x: x[1]
    )
    node_labels = [f"{name}  [{nid}]" for nid, name in all_nodes]
    node_ids    = [nid for nid, _ in all_nodes]

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        default_src = node_ids.index("4") if "4" in node_ids else 0
        src_label = st.selectbox("Emergency Origin", node_labels, index=default_src, key="em_src")
        src_id = node_ids[node_labels.index(src_label)]
    with c2:
        med_labels = [f"{n.name}  [{n.id}]" for n in medical]
        med_ids    = [n.id for n in medical]
        tgt_label  = st.selectbox("Target Hospital", med_labels, key="em_tgt")
        tgt_id     = med_ids[med_labels.index(tgt_label)]
    with c3:
        time_sel = st.selectbox("Time", ["morning","afternoon","evening","night"], key="em_time")

    if st.button("🚨  Dispatch Emergency Vehicle", use_container_width=True, key="em_go"):
        with st.spinner("Running algorithms..."):
            d_res = get_shortest_path(graph, src_id, tgt_id, time_sel)
            a_res = get_emergency_path(graph, src_id, tgt_id, time_sel)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        nodes_saved = d_res.nodes_explored - a_res.nodes_explored
        same_path   = d_res.path == a_res.path
        with k1: kpi_card("A* Travel Time", f"{a_res.total_time:.1f}m", "Emergency route", "#ff3366", "🚑")
        with k2: kpi_card("Nodes Explored (A*)", str(a_res.nodes_explored), f"vs {d_res.nodes_explored} (Dijkstra)", "#00d4ff", "🔍")
        with k3: kpi_card("Nodes Saved", str(nodes_saved), "A* efficiency gain", "#00ff88" if nodes_saved > 0 else "#ffcc00", "⚡")
        with k4: kpi_card("Same Path?", "YES ✓" if same_path else "NO ✗", "Optimality check", "#00ff88" if same_path else "#ff3366", "✅")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        map_c, bar_c = st.columns([3, 1])

        with map_c:
            fig = build_map(graph, time_of_day=time_sel, route_path=a_res.path)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with bar_c:
            section_header("Algorithm Race", "#ff3366")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Dijkstra", "A*"],
                y=[d_res.nodes_explored, a_res.nodes_explored],
                marker_color=["#0066ff", "#ff3366"],
                text=[str(d_res.nodes_explored), str(a_res.nodes_explored)],
                textposition="outside",
                textfont=dict(color="#c8e0f4", family="JetBrains Mono", size=12),
            ))
            fig2.update_layout(
                paper_bgcolor="#060b14", plot_bgcolor="#0a1628",
                height=220, margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(color="#4a7a9b", tickfont=dict(family="JetBrains Mono", size=10)),
                yaxis=dict(color="#4a7a9b", gridcolor="#1a3a5c",
                           tickfont=dict(family="JetBrains Mono", size=10)),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            section_header("A* Route", "#ff3366")
            if a_res.found:
                for name in a_res.path_names(graph):
                    st.markdown(f"""
                    <div style='
                        font-family:"JetBrains Mono",monospace;
                        font-size:0.72rem; color:#c8e0f4;
                        padding:0.15rem 0;
                    '>→ {name}</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: INFRASTRUCTURE
# ─────────────────────────────────────────────

def page_infrastructure(graph):
    page_header("Infrastructure", "MST expansion · Transit DP · Maintenance DP", "🏗️")

    tabs = st.tabs(["🛣️  Road Expansion (MST)", "🚌  Bus Allocation (DP)", "🔧  Maintenance (DP)"])

    with tabs[0]:
        section_header("Kruskal's MST — Optimal Road Expansion", "#00ff88")
        if st.button("▶  Run MST Analysis", key="mst_run"):
            from kruskal_mst import plan_new_infrastructure, visualize_infrastructure
            with st.spinner("Running Kruskal's..."):
                result = plan_new_infrastructure(graph)
                fig_mst = visualize_infrastructure(graph, result["chosen_roads"])

            k1, k2, k3 = st.columns(3)
            with k1: kpi_card("New Roads", str(len(result["chosen_roads"])), "Selected by MST", "#00ff88", "🛣️")
            with k2: kpi_card("Total Cost", f"{result['total_cost']:.0f}M", "EGP construction", "#ffcc00", "💰")
            with k3: kpi_card("Priority Savings", f"{result['savings']:.0f}M", "Discount applied", "#00d4ff", "⚡")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # Convert matplotlib fig to Streamlit
            import io
            buf = io.BytesIO()
            fig_mst.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                            facecolor="#060b14")
            buf.seek(0)
            st.image(buf, use_container_width=True)

            section_header("Selected Roads", "#00ff88")
            rows = []
            for e in result["chosen_roads"]:
                fn = graph.get_node(e.from_id)
                tn = graph.get_node(e.to_id)
                rows.append({
                    "From": fn.name if fn else e.from_id,
                    "To":   tn.name if tn else e.to_id,
                    "Distance (km)": e.distance,
                    "Cost (M EGP)":  e.cost,
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "Cost (M EGP)": st.column_config.NumberColumn(format="%.0f"),
                    }
                )

    with tabs[1]:
        section_header("Dynamic Programming — Bus Fleet Allocation", "#aa44ff")
        if st.button("▶  Run Transit Optimizer", key="transit_run"):
            from public_transit_scheduler import optimize_transit, plot_transit
            with st.spinner("Solving DP..."):
                result = optimize_transit()

            k1, k2, k3 = st.columns(3)
            with k1: kpi_card("Fleet Size", "203", "Total buses deployed", "#aa44ff", "🚌")
            with k2: kpi_card("Avg Coverage", f"{result.fleet_coverage:.1%}", "Daily demand served", "#00d4ff", "📊")
            with k3: kpi_card("Passengers Served", f"{result.total_served/1e6:.2f}M", "Per day", "#00ff88", "👥")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            import io
            fig_transit = plot_transit(result)
            buf = io.BytesIO()
            fig_transit.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                                facecolor="#060b14")
            buf.seek(0)
            st.image(buf, use_container_width=True)

    with tabs[2]:
        section_header("DP Knapsack — Road Maintenance Optimizer", "#ff8800")
        budget = st.slider("Repair Budget (cost-units)", 500, 5000, 2000, 100, key="maint_budget")
        if st.button("▶  Run Maintenance Optimizer", key="maint_run"):
            from dp_maintenance import load_edges, load_nodes, solve_maintenance, plot_condition_heatmap
            with st.spinner("Solving knapsack DP..."):
                edges = load_edges(os.path.join(BASE_DIR, "edges.csv"))
                nodes = load_nodes(os.path.join(BASE_DIR, "nodes.csv"))
                result = solve_maintenance(edges, budget)

            k1, k2, k3 = st.columns(3)
            with k1: kpi_card("Roads Selected", str(len(result.selected_roads)), "For repair", "#ff8800", "🔧")
            with k2: kpi_card("Budget Used", f"{result.total_cost:,}", f"of {budget:,} units", "#ffcc00", "💰")
            with k3: kpi_card("Score Gain", f"+{result.total_value}", "Condition points", "#00ff88", "📈")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            import io
            fig_heat = plot_condition_heatmap(edges, nodes, result)
            buf = io.BytesIO()
            fig_heat.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                             facecolor="#060b14")
            buf.seek(0)
            st.image(buf, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: SYSTEM DASHBOARD
# ─────────────────────────────────────────────

def page_dashboard(graph):
    page_header("System Dashboard", "Network health · Congestion heatmap · Signal optimizer", "📊")

    time_sel = st.selectbox("Time of Day", ["morning","afternoon","evening","night"],
                            key="dash_time", label_visibility="collapsed")

    # ── Network overview metrics ────────────────────────────────
    section_header("Network Overview")
    existing = graph.get_existing_edges()

    def _delay_ratio(edge, tod):
        # (travel_time - free_flow) / free_flow  capped at 1.0
        # free_flow = distance km at 60 km/h = distance minutes
        free = edge.distance
        if free <= 0:
            return 0.0
        actual = graph.get_weight(edge.from_id, edge.to_id, tod)
        return min(max((actual - free) / (free * 4), 0.0), 1.0)

    cong_vals = [_delay_ratio(e, time_sel) for e in existing]
    avg_cong  = sum(cong_vals) / len(cong_vals) if cong_vals else 0
    high_cong = sum(1 for c in cong_vals if c > 0.35)
    severe    = sum(1 for c in cong_vals if c > 0.65)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("Total Nodes",    str(len(graph.nodes)),         "In network",        "#00d4ff", "🏙️")
    with k2: kpi_card("Active Roads",   str(len(existing)),            "Existing edges",    "#00ff88", "🛣️")
    with k3: kpi_card("Avg Congestion", f"{avg_cong*100:.0f}%",        time_sel.title(),    "#ffcc00", "🚦")
    with k4: kpi_card("High Congestion",str(high_cong),                "Roads > 65%",       "#ff6600", "⚠️")
    with k5: kpi_card("Severe",         str(severe),                   "Roads > 85%",       "#ff3366", "🚨")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Congestion bar chart ────────────────────────────────────
    left_col, right_col = st.columns([2, 1])

    with left_col:
        section_header("Per-Road Congestion", "#ffcc00")
        rows = []
        for e, c in zip(existing, cong_vals):
            n_f = graph.get_node(e.from_id)
            n_t = graph.get_node(e.to_id)
            rows.append({
                "Road": f"{n_f.name[:14]} → {n_t.name[:14]}",
                "Congestion": round(c * 100, 1),
                "Level": _congestion_label(c),
            })
        df = pd.DataFrame(rows).sort_values("Congestion", ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=df["Congestion"],
            y=df["Road"],
            orientation="h",
            marker=dict(
                color=df["Congestion"],
                colorscale=[[0,"#00ff88"],[0.4,"#ffcc00"],[0.7,"#ff6600"],[1,"#ff0033"]],
                cmin=0, cmax=100,
            ),
            text=[f"{v:.0f}%" for v in df["Congestion"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=9, color="#8ba8cc"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="#060b14", plot_bgcolor="#0a1628",
            height=min(80 + len(df) * 22, 520),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(color="#4a7a9b", gridcolor="#1a3a5c", ticksuffix="%",
                       tickfont=dict(family="JetBrains Mono", size=9)),
            yaxis=dict(color="#c8e0f4", tickfont=dict(family="JetBrains Mono", size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with right_col:
        section_header("Signal Optimizer", "#00d4ff")
        intersection_nodes = [
            (nid, graph.get_node(nid).name)
            for nid in graph.nodes
            if len(graph.get_edges_from(nid)) >= 3
        ]
        if intersection_nodes:
            choices = [f"{name}  [{nid}]" for nid, name in intersection_nodes]
            ids     = [nid for nid, _ in intersection_nodes]
            sel_label = st.selectbox("Intersection", choices, key="sig_int")
            sel_id    = ids[choices.index(sel_label)]
            steps     = st.slider("Simulation Steps", 5, 20, 10, key="sig_steps")

            if st.button("▶  Run Signal Sim", key="sig_run", use_container_width=True):
                from greedy import SignalController, plot_signal_dashboard
                with st.spinner("Simulating..."):
                    ctrl   = SignalController(graph, sel_id, time_sel)
                    result = ctrl.run_simulation(steps=steps)

                kpi_card("Total Cars Saved", str(result.total_improvement),
                         f"Over {steps} steps", "#00d4ff", "🚦")
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                kpi_card("Emergency Pre-emptions", str(result.emergency_count),
                         "Priority overrides", "#ff3366", "🚨")
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                import io
                fig_sig = plot_signal_dashboard(result)
                buf = io.BytesIO()
                fig_sig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                                facecolor="#060b14")
                buf.seek(0)
                st.image(buf, use_container_width=True)

    # ── Node population treemap ─────────────────────────────────
    section_header("Neighbourhood Population Map", "#aa44ff")
    pop_nodes = [n for n in graph.nodes.values() if n.population > 0]
    if pop_nodes:
        fig_tree = go.Figure(go.Treemap(
            labels=[n.name for n in pop_nodes],
            parents=["" for _ in pop_nodes],
            values=[n.population for n in pop_nodes],
            marker=dict(
                colorscale=[[0,"#0a1628"],[0.5,"#0066ff"],[1,"#00d4ff"]],
                showscale=False,
            ),
            textfont=dict(family="JetBrains Mono", size=11, color="#e0f0ff"),
            hovertemplate="<b>%{label}</b><br>Population: %{value:,}<extra></extra>",
        ))
        fig_tree.update_layout(
            paper_bgcolor="#060b14",
            margin=dict(l=0, r=0, t=0, b=0),
            height=280,
        )
        st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Load graph (cached)
    with st.spinner("Initialising Cairo Traffic AI..."):
        graph = load_graph()

    # Sidebar — returns user selections
    src_id, tgt_id, time_sel, algo_sel, find_clicked = render_sidebar(graph)

    # Route page
    page = st.session_state.active_page
    if page == "Trip Planner":
        page_trip_planner(graph, src_id, tgt_id, time_sel, algo_sel, find_clicked)
    elif page == "Emergency Response":
        page_emergency(graph)
    elif page == "Infrastructure":
        page_infrastructure(graph)
    elif page == "System Dashboard":
        page_dashboard(graph)


if __name__ == "__main__":
    main()