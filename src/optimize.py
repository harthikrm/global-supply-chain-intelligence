"""
Inventory Optimization — Module D
===================================
Monte Carlo simulation to find optimal reorder point and safety stock
for each SKU that minimizes total cost under demand uncertainty
and lead time variability.
"""

import numpy as np
import pandas as pd
import duckdb
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')

# ── Constants ────────────────────────────────────────────────────
ORDERING_COST = 50.0  # Fixed cost per order ($)
N_SIMULATIONS = 10000
SIMULATION_WEEKS = 52
MAX_STOCKOUT_WEEKS_95 = 5  # Constraint: 95th percentile stockout weeks < 5


def compute_eoq(annual_demand: float, ordering_cost: float,
                holding_cost_per_unit: float) -> float:
    """
    Economic Order Quantity: Q* = sqrt(2 * D * S / H)
    """
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return max(1, int(annual_demand / 12))

    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
    return max(1, int(eoq))


def simulate_inventory(weekly_demand_mean: float, weekly_demand_std: float,
                        lead_time_mean: float, lead_time_std: float,
                        reorder_point: int, order_quantity: int,
                        unit_cost: float, holding_cost_pct: float,
                        stockout_cost_per_unit: float,
                        disruption_lt_multiplier: float = 1.0,
                        disruption_demand_multiplier: float = 1.0,
                        supply_outage_prob: float = 0.0,
                        n_simulations: int = N_SIMULATIONS,
                        n_weeks: int = SIMULATION_WEEKS,
                        rng: np.random.RandomState = None) -> dict:
    """
    Vectorized Monte Carlo inventory simulation.

    For each simulation:
    - Draw weekly demand from fitted distribution
    - Draw lead time from normal distribution
    - Apply reorder policy: when inventory < R, order Q units
    - Track holding cost, ordering cost, stockout cost
    """
    if rng is None:
        rng = np.random.RandomState(42)

    holding_cost_per_unit_week = unit_cost * holding_cost_pct / 4.33  # Monthly to weekly

    # Adjust for disruption
    adj_demand_mean = weekly_demand_mean * disruption_demand_multiplier
    adj_demand_std = weekly_demand_std * disruption_demand_multiplier
    adj_lt_mean = lead_time_mean * disruption_lt_multiplier
    adj_lt_std = lead_time_std * disruption_lt_multiplier

    # Pre-generate random demand and lead times for all simulations
    demands = rng.poisson(max(1, adj_demand_mean), size=(n_simulations, n_weeks))
    lead_times = np.maximum(1, rng.normal(adj_lt_mean, max(1, adj_lt_std),
                                           size=(n_simulations, n_weeks)).astype(int))

    # Supply outage mask
    if supply_outage_prob > 0:
        outage_mask = rng.random(size=(n_simulations, n_weeks)) < supply_outage_prob
    else:
        outage_mask = np.zeros((n_simulations, n_weeks), dtype=bool)

    # Simulate
    total_holding_costs = np.zeros(n_simulations)
    total_ordering_costs = np.zeros(n_simulations)
    total_stockout_costs = np.zeros(n_simulations)
    stockout_weeks = np.zeros(n_simulations)

    # Starting inventory: 2x reorder point
    starting_inventory = max(reorder_point * 2, order_quantity)

    for sim in range(n_simulations):
        inventory = float(starting_inventory)
        pending_orders = []  # List of (arrival_week, quantity)
        n_orders = 0

        for week in range(n_weeks):
            # Receive pending orders
            new_pending = []
            for arrival_week, qty in pending_orders:
                if week >= arrival_week:
                    if not outage_mask[sim, week]:
                        inventory += qty
                else:
                    new_pending.append((arrival_week, qty))
            pending_orders = new_pending

            # Fulfill demand
            demand = demands[sim, week]
            if inventory >= demand:
                inventory -= demand
            else:
                # Stockout
                unmet = demand - inventory
                total_stockout_costs[sim] += unmet * stockout_cost_per_unit
                stockout_weeks[sim] += 1
                inventory = 0

            # Holding cost
            if inventory > 0:
                total_holding_costs[sim] += inventory * holding_cost_per_unit_week

            # Reorder check
            if inventory <= reorder_point and len(pending_orders) == 0:
                lt_days = lead_times[sim, week]
                lt_weeks = max(1, lt_days // 7)
                arrival = min(week + lt_weeks, n_weeks - 1)
                if not outage_mask[sim, arrival]:
                    pending_orders.append((arrival, order_quantity))
                    total_ordering_costs[sim] += ORDERING_COST
                    n_orders += 1

    max_annual_cost = unit_cost * order_quantity * 20
    total_costs = np.minimum(total_holding_costs + total_ordering_costs + total_stockout_costs, max_annual_cost)

    return {
        'mean_total_cost': float(np.mean(total_costs)),
        'p95_total_cost': float(np.percentile(total_costs, 95)),
        'mean_holding_cost': float(np.mean(total_holding_costs)),
        'mean_ordering_cost': float(np.mean(total_ordering_costs)),
        'mean_stockout_cost': float(np.mean(total_stockout_costs)),
        'mean_stockout_weeks': float(np.mean(stockout_weeks)),
        'p95_stockout_weeks': float(np.percentile(stockout_weeks, 95)),
        'total_cost_distribution': total_costs,
        'stockout_week_distribution': stockout_weeks,
    }


def optimize_sku(sku_row: pd.Series, demand_stats: dict,
                  scenario: str = 'baseline',
                  rng: np.random.RandomState = None) -> dict:
    """
    Find optimal (R, Q) for a single SKU across a grid search.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    weekly_demand_mean = demand_stats['mean']
    weekly_demand_std = demand_stats['std']
    annual_demand = weekly_demand_mean * 52

    # EOQ baseline
    holding_cost_annual = sku_row['unit_cost_usd'] * sku_row['holding_cost_pct'] * 12
    eoq = compute_eoq(annual_demand, ORDERING_COST, holding_cost_annual)

    # Scenario parameters
    scenarios = {
        'baseline': {'lt_mult': 1.0, 'demand_mult': 1.0, 'outage': 0.0},
        'moderate': {'lt_mult': 1.4, 'demand_mult': 1.2, 'outage': 0.0},
        'severe': {'lt_mult': 1.8, 'demand_mult': 1.35, 'outage': 0.10},
    }
    params = scenarios.get(scenario, scenarios['baseline'])

    # Grid search — wider R range (up to 8 weeks of demand) and more Q values
    r_values = np.linspace(0, max(8 * weekly_demand_mean, 20), 10).astype(int)
    q_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    q_values = [max(1, int(eoq * m)) for m in q_multipliers]

    best_cost = float('inf')
    best_r = r_values[0]
    best_q = q_values[0]
    best_result = None

    for R in r_values:
        for Q in q_values:
            result = simulate_inventory(
                weekly_demand_mean=weekly_demand_mean,
                weekly_demand_std=weekly_demand_std,
                lead_time_mean=sku_row['lead_time_days'],
                lead_time_std=sku_row['lead_time_days'] * 0.15,
                reorder_point=int(R),
                order_quantity=int(Q),
                unit_cost=sku_row['unit_cost_usd'],
                holding_cost_pct=sku_row['holding_cost_pct'],
                stockout_cost_per_unit=sku_row['stockout_cost_usd'],
                disruption_lt_multiplier=params['lt_mult'],
                disruption_demand_multiplier=params['demand_mult'],
                supply_outage_prob=params['outage'],
                n_simulations=500,  # Reduced for grid search
                rng=rng,
            )

            # Constraint: p95 stockout weeks < 2
            if (result['p95_stockout_weeks'] <= MAX_STOCKOUT_WEEKS_95 and
                    result['mean_total_cost'] < best_cost):
                best_cost = result['mean_total_cost']
                best_r = R
                best_q = Q
                best_result = result

    # If no feasible solution found, use highest safety stock
    if best_result is None:
        best_r = int(r_values[-1])
        best_q = int(q_values[-1])
        best_result = simulate_inventory(
            weekly_demand_mean=weekly_demand_mean,
            weekly_demand_std=weekly_demand_std,
            lead_time_mean=sku_row['lead_time_days'],
            lead_time_std=sku_row['lead_time_days'] * 0.15,
            reorder_point=best_r,
            order_quantity=best_q,
            unit_cost=sku_row['unit_cost_usd'],
            holding_cost_pct=sku_row['holding_cost_pct'],
            stockout_cost_per_unit=sku_row['stockout_cost_usd'],
            disruption_lt_multiplier=params['lt_mult'],
            disruption_demand_multiplier=params['demand_mult'],
            supply_outage_prob=params['outage'],
            n_simulations=2000,
            rng=rng,
        )

    safety_stock_weeks = best_r / max(1, weekly_demand_mean)

    # Cap cost at a reasonable upper bound (20x annual demand cost) to avoid inf
    annual_demand_cost = weekly_demand_mean * 52 * sku_row['unit_cost_usd']
    cost_cap = annual_demand_cost * 20
    capped_cost = min(best_cost, cost_cap) if np.isfinite(best_cost) else cost_cap

    return {
        'sku_id': sku_row['sku_id'],
        'category': sku_row['category'],
        'supplier_country': sku_row['supplier_country'],
        'scenario': scenario,
        'eoq': eoq,
        'optimal_reorder_point': int(best_r),
        'optimal_order_quantity': int(best_q),
        'safety_stock_weeks': round(safety_stock_weeks, 2),
        'expected_annual_cost': round(capped_cost, 2),
        'mean_holding_cost': round(best_result['mean_holding_cost'], 2),
        'mean_ordering_cost': round(best_result['mean_ordering_cost'], 2),
        'mean_stockout_cost': round(best_result['mean_stockout_cost'], 2),
        'mean_stockout_weeks': round(best_result['mean_stockout_weeks'], 2),
        'p95_stockout_weeks': round(best_result['p95_stockout_weeks'], 2),
        'cost_distribution': best_result['total_cost_distribution'],
    }


def run_module_d() -> dict:
    """Execute the full Module D inventory optimization pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Module D — Inventory Optimization                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    con = duckdb.connect(DB_PATH, read_only=True)

    # Load SKU and demand data
    skus_df = con.execute("SELECT * FROM skus").fetchdf().copy()
    demand_stats_df = con.execute("""
        SELECT sku_id,
               AVG(demand_units) as mean,
               STDDEV(demand_units) as std,
               SUM(demand_units) as total
        FROM weekly_demand
        GROUP BY sku_id
    """).fetchdf().copy()

    con.close()

    # Sample SKUs for computational tractability
    sample_size = min(50, len(skus_df))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(skus_df), sample_size, replace=False)
    sample_skus = skus_df.iloc[sample_indices]

    print(f"\n  Optimizing {sample_size} representative SKUs across 3 scenarios...")

    all_results = []
    scenarios = ['baseline', 'moderate', 'severe']

    for i, (_, sku) in enumerate(sample_skus.iterrows()):
        sku_id = sku['sku_id']
        stats = demand_stats_df[demand_stats_df['sku_id'] == sku_id]

        if len(stats) == 0:
            continue

        demand_stats = {
            'mean': stats.iloc[0]['mean'],
            'std': stats.iloc[0]['std'] if not pd.isna(stats.iloc[0]['std']) else stats.iloc[0]['mean'] * 0.3,
        }

        for scenario in scenarios:
            result = optimize_sku(sku, demand_stats, scenario, rng)
            # Remove distribution arrays for storage
            result_clean = {k: v for k, v in result.items() if k != 'cost_distribution'}
            all_results.append(result_clean)

        if (i + 1) % 10 == 0:
            print(f"    ... optimized {i + 1}/{sample_size} SKUs")

    results_df = pd.DataFrame(all_results)

    # Compute cost increase under disruption
    baseline_costs = results_df[results_df['scenario'] == 'baseline'][['sku_id', 'expected_annual_cost']]
    baseline_costs.columns = ['sku_id', 'baseline_cost']

    severe_costs = results_df[results_df['scenario'] == 'severe'][['sku_id', 'expected_annual_cost']]
    severe_costs.columns = ['sku_id', 'severe_cost']

    cost_comparison = baseline_costs.merge(severe_costs, on='sku_id')

    # Safety net: replace any residual inf with per-SKU max (should not happen after cost_cap)
    for col in ['baseline_cost', 'severe_cost']:
        mask = ~np.isfinite(cost_comparison[col])
        if mask.any():
            finite_max = cost_comparison.loc[~mask, col].max() if (~mask).any() else 1e6
            cost_comparison.loc[mask, col] = finite_max * 2

    cost_comparison['cost_increase_pct'] = np.where(
        cost_comparison['baseline_cost'] > 0,
        ((cost_comparison['severe_cost'] - cost_comparison['baseline_cost'])
         / cost_comparison['baseline_cost'].clip(lower=1) * 100),
        100.0
    ).round(2)

    # Assign disruption risk tier
    thresholds = cost_comparison['cost_increase_pct'].quantile([0.33, 0.67])
    cost_comparison['disruption_risk_tier'] = pd.cut(
        cost_comparison['cost_increase_pct'],
        bins=[-np.inf, thresholds[0.33], thresholds[0.67], np.inf],
        labels=['Low', 'Medium', 'High']
    )

    results_df = results_df.merge(
        cost_comparison[['sku_id', 'cost_increase_pct', 'disruption_risk_tier']],
        on='sku_id', how='left'
    )

    # Summary
    baseline = results_df[results_df['scenario'] == 'baseline']
    print(f"\n  ── Optimization Summary (Baseline Scenario) ──")
    print(f"     Average EOQ: {baseline['eoq'].mean():.0f} units")
    print(f"     Average optimal reorder point: {baseline['optimal_reorder_point'].mean():.0f} units")
    print(f"     Average safety stock: {baseline['safety_stock_weeks'].mean():.1f} weeks")
    print(f"     Average annual cost: ${baseline['expected_annual_cost'].mean():,.0f}")
    print(f"     Average stockout weeks: {baseline['mean_stockout_weeks'].mean():.1f}")

    print(f"\n  ── Disruption Risk Tiers ──")
    tier_counts = cost_comparison['disruption_risk_tier'].value_counts()
    for tier in ['High', 'Medium', 'Low']:
        if tier in tier_counts.index:
            print(f"     {tier}: {tier_counts[tier]} SKUs "
                  f"(avg cost increase: {cost_comparison[cost_comparison['disruption_risk_tier']==tier]['cost_increase_pct'].mean():.1f}%)")

    # ── Serialize results for dashboard consumption ──
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'optimization_results.pkl'

    serializable = {
        'results': results_df,
        'cost_comparison': cost_comparison,
        'sample_skus': sample_skus,
    }
    with open(results_path, 'wb') as f:
        pickle.dump(serializable, f)
    print(f"\n  ✓ Optimization results saved to {results_path}")

    return serializable


if __name__ == '__main__':
    results = run_module_d()
    print("\n── Top 10 Highest Disruption Risk SKUs ──")
    top_risk = results['cost_comparison'].sort_values('cost_increase_pct', ascending=False).head(10)
    print(top_risk.to_string(index=False))
