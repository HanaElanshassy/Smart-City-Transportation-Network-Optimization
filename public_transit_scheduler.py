"""
=============================================================
  CSE112 - Member 5: Public Transit Scheduler (DP)
  Cairo Smart City Transportation Network Optimization
=============================================================

Optimizes bus allocation to maximize total weighted demand
coverage across routes B1-B10 using a 203-bus fleet.

Root-cause fix vs. original:
    The original DP dumped all 183 remaining buses into B1
    because demand (35 000+) vastly exceeds what 203 buses
    (capacity_per_bus=50) can serve.  Maximising raw
    passengers-served always collapses to one route.

    Fix: the DP value unit is now "demand coverage score" —
    a diminishing-returns function that rewards spreading
    buses proportionally across routes:
        score(k buses, route i) = sqrt(k / buses_needed_i) * 100
    This is still a standard bounded-knapsack DP; only the
    objective function changes.

Public API:
    result = optimize_transit()          # returns TransitResult
    fig    = plot_transit(result)        # matplotlib Figure
"""

import os
import sys
import math
import matplotlib
matplotlib.use('Agg')   # non-interactive — safe for servers / Vercel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ROUTES           = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]
DEMAND           = [35000, 42000, 28000, 31000, 25000, 33000, 21000, 17000, 39000, 28000]
TOTAL_BUSES      = 203
CAPACITY_PER_BUS = 50      # passengers per bus per trip
MIN_BUSES        = 2       # fairness floor: every route gets at least 2


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

class TransitResult:
    def __init__(self, allocation, demand, routes, capacity_per_bus):
        self.allocation       = allocation           # list[int]  buses per route
        self.demand           = demand               # list[int]  passengers/day
        self.routes           = routes               # list[str]
        self.capacity_per_bus = capacity_per_bus

        self.served     = [min(a * capacity_per_bus, d)
                           for a, d in zip(allocation, demand)]
        self.coverage   = [s / d for s, d in zip(self.served, demand)]
        self.total_served   = sum(self.served)
        self.total_demand   = sum(demand)
        self.fleet_coverage = self.total_served / self.total_demand

    def summary(self):
        """Print a formatted allocation report."""
        print("\n" + "═" * 62)
        print("  🚌  SMART CITY BUS ALLOCATION REPORT  (DP)")
        print("═" * 62)
        print(f"  Fleet size          : {TOTAL_BUSES} buses")
        print(f"  Capacity per bus    : {self.capacity_per_bus} passengers")
        print(f"  Total demand        : {self.total_demand:,} passengers/day")
        print(f"  Total served        : {self.total_served:,} passengers/day")
        print(f"  Fleet coverage      : {self.fleet_coverage:.1%}")
        print("─" * 62)
        print(f"  {'Route':<6} {'Buses':>6} {'Demand':>8} {'Served':>8} {'Coverage':>9}  Bar")
        print(f"  {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*9}  {'─'*20}")
        for i, route in enumerate(self.routes):
            cov  = self.coverage[i]
            bar  = "█" * int(cov * 20)
            flag = "🔴" if cov < 0.05 else ("🟡" if cov < 0.15 else "🟢")
            print(f"  {route:<6} {self.allocation[i]:>6} {self.demand[i]:>8,} "
                  f"{self.served[i]:>8,} {cov:>8.1%}  {flag} {bar}")
        print("═" * 62)


# ─────────────────────────────────────────────
# DP SOLVER
# ─────────────────────────────────────────────

def _coverage_score(buses: int, demand: int) -> float:
    """
    Diminishing-returns coverage score for `buses` assigned to a
    route with `demand` passengers.

    Uses sqrt to model the law of diminishing returns: the first
    few buses on an underserved route matter most.  Scaled to
    100 so the DP table stores integers (we multiply by 100 and
    round), keeping the table small.

        score = sqrt(buses / buses_needed) * 100

    This is admissible: proportional, bounded, and guarantees
    the DP spreads buses across routes rather than stacking them.
    """
    if buses == 0:
        return 0.0
    buses_needed = math.ceil(demand / CAPACITY_PER_BUS)
    return math.sqrt(buses / buses_needed) * 100


def optimize_transit() -> TransitResult:
    """
    Run the DP bus-allocation optimiser.

    Algorithm: bounded 0/1 knapsack variant.
        State  : dp[i][j] = best total coverage score using the
                            first i routes and j extra buses
        Action : assign k ∈ [0, j] extra buses to route i
        Value  : coverage_score(k + MIN_BUSES, demand[i])

    Time  complexity: O(n × B²)  where B = remaining_buses = 183
    Space complexity: O(n × B)   (two 2D arrays)

    Returns:
        TransitResult with final allocation and metrics
    """
    n               = len(ROUTES)
    remaining_buses = TOTAL_BUSES - MIN_BUSES * n   # 183

    # ── DP table (scores scaled × 100, stored as float) ──────────
    # dp[i][j] = best score assigning j extra buses across routes 0..i-1
    dp     = [[0.0] * (remaining_buses + 1) for _ in range(n + 1)]
    choice = [[0]   * (remaining_buses + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        d = DEMAND[i - 1]
        for j in range(remaining_buses + 1):
            # Try every possible extra-bus count k for route i
            dp[i][j]     = dp[i - 1][j]   # default: give route i zero extra
            choice[i][j] = 0
            for k in range(1, j + 1):
                score = dp[i - 1][j - k] + _coverage_score(k + MIN_BUSES, d)
                if score > dp[i][j]:
                    dp[i][j]     = score
                    choice[i][j] = k

    # ── Backtrack to recover allocation ──────────────────────────
    allocation = [MIN_BUSES] * n
    j = remaining_buses
    for i in range(n, 0, -1):
        extra          = choice[i][j]
        allocation[i - 1] += extra
        j              -= extra

    return TransitResult(allocation, DEMAND, ROUTES, CAPACITY_PER_BUS)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_transit(result: TransitResult) -> plt.Figure:
    """
    Generate a two-panel bar chart showing:
        Left  — bus allocation per route
        Right — demand coverage % per route

    Returns a matplotlib Figure (caller saves or displays it).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x     = np.arange(len(result.routes))
    width = 0.6

    # ── Panel 1: Bus allocation ───────────────────────────────────
    colors1 = ['#2ecc71' if c >= 0.15 else ('#f39c12' if c >= 0.05 else '#e74c3c')
               for c in result.coverage]
    bars1 = ax1.bar(x, result.allocation, width, color=colors1, edgecolor='black', linewidth=0.6)
    ax1.axhline(TOTAL_BUSES / len(ROUTES), color='navy', linestyle='--',
                linewidth=1.2, label=f'Fleet average ({TOTAL_BUSES/len(ROUTES):.1f})')
    ax1.set_title("Bus Allocation per Route", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Route")
    ax1.set_ylabel("Buses Assigned")
    ax1.set_xticks(x)
    ax1.set_xticklabels(result.routes)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars1, result.allocation):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Panel 2: Demand coverage % ────────────────────────────────
    cov_pct = [c * 100 for c in result.coverage]
    colors2 = ['#2ecc71' if c >= 15 else ('#f39c12' if c >= 5 else '#e74c3c')
               for c in cov_pct]
    bars2 = ax2.bar(x, cov_pct, width, color=colors2, edgecolor='black', linewidth=0.6)
    ax2.axhline(result.fleet_coverage * 100, color='navy', linestyle='--',
                linewidth=1.2, label=f'Avg coverage ({result.fleet_coverage:.1%})')
    ax2.set_title("Demand Coverage per Route (%)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Route")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(result.routes)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, cov_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Legend ────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color='#2ecc71', label='Good  (≥15% coverage)'),
        mpatches.Patch(color='#f39c12', label='Fair  (5–15% coverage)'),
        mpatches.Patch(color='#e74c3c', label='Low   (<5% coverage)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        f"Cairo Smart City — Optimal Bus Allocation  "
        f"(203 buses | {result.fleet_coverage:.1%} avg coverage)",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = optimize_transit()
    result.summary()

    fig = plot_transit(result)
    out = os.path.join(BASE_DIR, "transit_allocation.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Chart saved → {out}")
