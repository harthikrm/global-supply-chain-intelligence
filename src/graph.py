"""
Supply Chain Network Graph — Module A
======================================
Builds a directed weighted graph of the supply chain network using
NetworkX. Computes centrality metrics, runs disruption simulations,
and generates Plotly visualizations.
"""

import numpy as np
import pandas as pd
import networkx as nx
import duckdb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')

# ── Synthetic Manufacturers ──────────────────────────────────────
MANUFACTURERS = {
    'Toyota_JP': {'country': 'Japan', 'categories': ['Auto Parts'], 'revenue': 2.8e11},
    'VW_DE': {'country': 'Germany', 'categories': ['Auto Parts'], 'revenue': 2.9e11},
    'Samsung_KR': {'country': 'South Korea', 'categories': ['Electronics'], 'revenue': 2.4e11},
    'TSMC_TW': {'country': 'Taiwan', 'categories': ['Electronics'], 'revenue': 7.5e10},
    'Apple_US': {'country': 'USA', 'categories': ['Electronics'], 'revenue': 3.8e11},
    'BASF_DE': {'country': 'Germany', 'categories': ['Chemicals', 'Metals'], 'revenue': 8.7e10},
    'Pfizer_US': {'country': 'USA', 'categories': ['Pharmaceuticals'], 'revenue': 1.0e11},
    'Nestlé_CH': {'country': 'Switzerland', 'categories': ['Food & Beverage'], 'revenue': 9.4e10},
    'ArcelorMittal_LU': {'country': 'Luxembourg', 'categories': ['Metals'], 'revenue': 7.9e10},
    'Inditex_ES': {'country': 'Spain', 'categories': ['Textiles'], 'revenue': 3.5e10},
}

DISTRIBUTORS = ['US_East', 'US_West', 'EU_Central', 'Asia_Pacific', 'Latin_America']

RETAILER_CLUSTERS = [
    f"Retailer_{region}_{i}"
    for region in ['US', 'EU', 'Asia', 'LatAm']
    for i in range(1, 6)
]


def build_supply_chain_graph(con: duckdb.DuckDBPyConnection = None) -> nx.DiGraph:
    """
    Build a directed weighted supply chain graph with four node layers:
    Supplier → Manufacturer → Distributor → Retailer
    """
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)

    G = nx.DiGraph()

    # ── Load data ──
    skus_df = con.execute("SELECT * FROM skus").fetchdf()
    trade_df = con.execute("""
        SELECT hs_code, reporter_name, SUM(trade_value_usd) as total_value
        FROM trade_flows
        WHERE flow_type = 'Import'
        GROUP BY hs_code, reporter_name
    """).fetchdf()

    demand_df = con.execute("""
        SELECT s.category, s.supplier_country,
               SUM(wd.demand_units) as total_demand,
               AVG(wd.demand_units) as avg_demand
        FROM weekly_demand wd
        JOIN skus s ON wd.sku_id = s.sku_id
        GROUP BY s.category, s.supplier_country
    """).fetchdf()

    # ── Layer 1: Supplier Nodes ──
    supplier_nodes = skus_df.groupby(['supplier_country', 'category']).agg(
        sku_count=('sku_id', 'count'),
        avg_lead_time=('lead_time_days', 'mean'),
        avg_cost=('unit_cost_usd', 'mean'),
    ).reset_index()

    for _, row in supplier_nodes.iterrows():
        node_id = f"{row['supplier_country']}-{row['category']}"
        G.add_node(node_id,
                   node_type='supplier',
                   country=row['supplier_country'],
                   category=row['category'],
                   sku_count=row['sku_count'],
                   avg_lead_time=row['avg_lead_time'],
                   avg_cost=row['avg_cost'])

    # ── Layer 2: Manufacturer Nodes ──
    for mfr_id, info in MANUFACTURERS.items():
        G.add_node(mfr_id,
                   node_type='manufacturer',
                   country=info['country'],
                   categories=info['categories'],
                   revenue=info['revenue'])

        # Connect suppliers to manufacturers
        for cat in info['categories']:
            for _, sup in supplier_nodes.iterrows():
                if sup['category'] == cat:
                    sup_node = f"{sup['supplier_country']}-{cat}"
                    # Weight by trade value and proximity
                    weight = sup['sku_count'] * sup['avg_cost'] / 1000
                    lead_time_normalized = sup['avg_lead_time'] / 60  # Normalize to 0-1

                    G.add_edge(sup_node, mfr_id,
                               weight=weight,
                               trade_value=weight * 1e6,
                               lead_time_normalized=lead_time_normalized,
                               edge_type='supply')

    # ── Layer 3: Distributor Nodes ──
    distributor_regions = {
        'US_East': ['USA'],
        'US_West': ['USA'],
        'EU_Central': ['Germany', 'Luxembourg', 'Spain', 'Switzerland'],
        'Asia_Pacific': ['Japan', 'South Korea', 'Taiwan', 'China', 'India', 'Vietnam', 'Thailand'],
        'Latin_America': ['Mexico', 'Brazil'],
    }

    for dist_id in DISTRIBUTORS:
        G.add_node(dist_id,
                   node_type='distributor',
                   region=dist_id,
                   connected_countries=distributor_regions[dist_id])

        for mfr_id, info in MANUFACTURERS.items():
            # Connect based on geographic proximity
            mfr_country = info['country']
            region_countries = distributor_regions[dist_id]

            if mfr_country in region_countries:
                weight = info['revenue'] / 1e10
            else:
                weight = info['revenue'] / 5e10

            lead_time = 0.3 if mfr_country in region_countries else 0.7

            G.add_edge(mfr_id, dist_id,
                       weight=weight,
                       lead_time_normalized=lead_time,
                       edge_type='distribution')

    # ── Layer 4: Retailer Nodes ──
    for retailer_id in RETAILER_CLUSTERS:
        region = retailer_id.split('_')[1]
        G.add_node(retailer_id,
                   node_type='retailer',
                   region=region)

        # Connect to appropriate distributors
        region_to_dist = {
            'US': ['US_East', 'US_West'],
            'EU': ['EU_Central'],
            'Asia': ['Asia_Pacific'],
            'LatAm': ['Latin_America'],
        }

        for dist_id in region_to_dist.get(region, []):
            # Demand volume weight from synthetic data
            demand_weight = np.random.RandomState(hash(retailer_id) % 2**32).uniform(5, 20)
            G.add_edge(dist_id, retailer_id,
                       weight=demand_weight,
                       edge_type='retail')

    print(f"  ✓ Built supply chain graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute graph centrality metrics for every node:
    - degree_centrality, betweenness_centrality, pagerank, clustering_coefficient
    """
    # Compute metrics
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    pagerank = nx.pagerank(G, weight='weight', max_iter=200)

    # Clustering on undirected version
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    # Build DataFrame
    rows = []
    for node in G.nodes():
        node_data = G.nodes[node]
        rows.append({
            'node_id': node,
            'node_type': node_data.get('node_type', 'unknown'),
            'country': node_data.get('country', ''),
            'category': node_data.get('category', ''),
            'degree_centrality': round(degree.get(node, 0), 6),
            'betweenness_centrality': round(betweenness.get(node, 0), 6),
            'pagerank': round(pagerank.get(node, 0), 6),
            'clustering_coefficient': round(clustering.get(node, 0), 6),
        })

    df = pd.DataFrame(rows)

    # Assign risk tiers
    bc_75 = df['betweenness_centrality'].quantile(0.75)
    bc_90 = df['betweenness_centrality'].quantile(0.90)

    def risk_tier(bc):
        if bc >= bc_90:
            return 'Critical'
        elif bc >= bc_75:
            return 'High'
        elif bc > 0:
            return 'Medium'
        return 'Low'

    df['risk_tier'] = df['betweenness_centrality'].apply(risk_tier)

    print(f"  ✓ Computed centrality metrics for {len(df)} nodes")
    print(f"    Critical nodes: {len(df[df['risk_tier'] == 'Critical'])}")
    print(f"    High risk nodes: {len(df[df['risk_tier'] == 'High'])}")

    return df


def simulate_disruption(G: nx.DiGraph, nodes_to_remove: list,
                        edge_weight_reduction: dict = None,
                        event_name: str = "Disruption") -> dict:
    """
    Simulate a supply chain disruption by removing nodes and/or
    reducing edge weights. Returns impact analysis.

    Parameters:
    -----------
    G : nx.DiGraph - Original graph
    nodes_to_remove : list - Nodes to completely remove
    edge_weight_reduction : dict - {(src, dst): reduction_factor} where
                           reduction_factor is the fraction to keep (0.4 = 60% reduction)
    event_name : str - Name of the disruption event
    """
    G_sim = G.copy()

    # Baseline metrics
    baseline_nodes = G.number_of_nodes()
    baseline_edges = G.number_of_edges()

    # Handle edge weight reduction
    if edge_weight_reduction:
        for (src, dst), factor in edge_weight_reduction.items():
            if G_sim.has_edge(src, dst):
                G_sim[src][dst]['weight'] *= factor

    # Remove nodes
    removed_trade_value = 0
    for node in nodes_to_remove:
        if node in G_sim:
            # Sum trade value on connected edges
            for _, _, data in G_sim.edges(node, data=True):
                removed_trade_value += data.get('trade_value', data.get('weight', 0) * 1e6)
            for _, _, data in G_sim.in_edges(node, data=True):
                removed_trade_value += data.get('trade_value', data.get('weight', 0) * 1e6)
            G_sim.remove_node(node)

    # Post-disruption metrics
    post_nodes = G_sim.number_of_nodes()
    post_edges = G_sim.number_of_edges()

    # Connectivity analysis
    G_undirected = G_sim.to_undirected()
    components = list(nx.connected_components(G_undirected))
    largest_component = max(components, key=len) if components else set()

    disconnected_nodes = post_nodes - len(largest_component)
    disconnected_pct = disconnected_nodes / post_nodes * 100 if post_nodes > 0 else 0

    # Path length analysis
    try:
        avg_path_pre = nx.average_shortest_path_length(G.to_undirected())
    except nx.NetworkXError:
        avg_path_pre = float('inf')

    try:
        largest_subgraph = G_sim.subgraph(largest_component).to_undirected()
        avg_path_post = nx.average_shortest_path_length(largest_subgraph)
    except (nx.NetworkXError, ValueError):
        avg_path_post = float('inf')

    path_increase = ((avg_path_post - avg_path_pre) / avg_path_pre * 100
                     if avg_path_pre > 0 and avg_path_pre != float('inf') else 0)

    # Recompute centrality for remaining graph
    if len(G_sim) > 0:
        new_betweenness = nx.betweenness_centrality(G_sim, weight='weight')
        new_chokepoints = [n for n, v in new_betweenness.items()
                          if v > np.percentile(list(new_betweenness.values()), 90)]
    else:
        new_chokepoints = []

    result = {
        'event_name': event_name,
        'nodes_removed': len(nodes_to_remove),
        'edges_removed': baseline_edges - post_edges,
        'trade_value_at_risk_usd': removed_trade_value,
        'disconnected_nodes': disconnected_nodes,
        'disconnected_pct': round(disconnected_pct, 2),
        'avg_path_length_increase_pct': round(path_increase, 2),
        'new_chokepoints': new_chokepoints,
        'num_components': len(components),
        'disruption_impact_score': round(
            0.4 * disconnected_pct / 100 +
            0.3 * min(path_increase / 100, 1.0) +
            0.3 * min(removed_trade_value / 1e12, 1.0),
            4
        ),
    }

    print(f"\n  ── Disruption Simulation: {event_name} ──")
    print(f"     Nodes removed: {result['nodes_removed']}")
    print(f"     Disconnected nodes: {result['disconnected_nodes']} ({result['disconnected_pct']}%)")
    print(f"     Path length increase: {result['avg_path_length_increase_pct']}%")
    print(f"     Trade value at risk: ${result['trade_value_at_risk_usd']:,.0f}")
    print(f"     Impact score: {result['disruption_impact_score']}")

    return result


def run_historical_simulations(G: nx.DiGraph) -> pd.DataFrame:
    """
    Run disruption simulations for three historical events:
    1. Ukraine Conflict (2022)
    2. Red Sea Disruption (2023)
    3. Russia Sanctions (2022)
    """
    print("\n── Running Historical Disruption Simulations ──")

    results = []

    # 1. Ukraine Conflict
    ukraine_nodes = [n for n in G.nodes()
                     if 'Turkey' in n and any(c in n for c in ['Wheat', 'Metals', 'Neon', 'Food'])]
    # Also look for nodes matching Ukraine-related categories
    ukraine_remove = [n for n in G.nodes()
                      if any(kw in n for kw in ['Food', 'Chemicals', 'Metals'])
                      and any(c in n for c in ['Turkey'])]
    if not ukraine_remove:
        # Fallback: remove high-risk food/metals supplier nodes
        ukraine_remove = [n for n in G.nodes()
                          if G.nodes[n].get('node_type') == 'supplier'
                          and G.nodes[n].get('category') in ['Food & Beverage', 'Metals']
                          and G.nodes[n].get('country') in ['Turkey', 'India']][:3]

    r1 = simulate_disruption(G, ukraine_remove, event_name="Ukraine Conflict (2022)")
    results.append(r1)

    # 2. Red Sea Disruption — reduce Asia→US/EU route weights by 60%
    edge_reductions = {}
    for u, v, data in G.edges(data=True):
        u_data = G.nodes.get(u, {})
        v_data = G.nodes.get(v, {})
        u_country = u_data.get('country', '')

        asia_countries = {'China', 'Japan', 'South Korea', 'Taiwan', 'India', 'Vietnam', 'Thailand'}

        if u_country in asia_countries:
            if 'US' in v or 'EU' in v or v_data.get('region', '') in ['US_East', 'US_West', 'EU_Central']:
                edge_reductions[(u, v)] = 0.4  # Keep 40% = 60% reduction

    r2 = simulate_disruption(G, [], edge_weight_reduction=edge_reductions,
                             event_name="Red Sea Shipping Disruption (2023)")
    results.append(r2)

    # 3. Russia Sanctions
    russia_remove = [n for n in G.nodes()
                     if G.nodes[n].get('node_type') == 'supplier'
                     and G.nodes[n].get('category') in ['Metals', 'Chemicals']
                     and G.nodes[n].get('country') in ['Germany', 'Turkey']][:4]

    r3 = simulate_disruption(G, russia_remove, event_name="Russia Sanctions (2022)")
    results.append(r3)

    return pd.DataFrame(results)


def create_network_plotly_figure(G: nx.DiGraph, centrality_df: pd.DataFrame):
    """
    Create an interactive Plotly network visualization.
    Node size = betweenness centrality, color = risk tier.
    """
    import plotly.graph_objects as go

    # Layout using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Color map for risk tiers
    risk_colors = {
        'Critical': '#EF4444',   # Red
        'High': '#F59E0B',       # Amber
        'Medium': '#3B82F6',     # Blue
        'Low': '#10B981',        # Green
    }

    node_type_shapes = {
        'supplier': 'circle',
        'manufacturer': 'diamond',
        'distributor': 'square',
        'retailer': 'triangle-up',
    }

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
        hoverinfo='none',
        mode='lines',
        name='Supply Links'
    )

    # Build node traces by risk tier
    node_traces = []
    centrality_dict = centrality_df.set_index('node_id').to_dict('index')

    for risk_tier, color in risk_colors.items():
        tier_nodes = centrality_df[centrality_df['risk_tier'] == risk_tier]
        x_vals, y_vals, sizes, texts, hovers = [], [], [], [], []

        for _, row in tier_nodes.iterrows():
            node_id = row['node_id']
            if node_id in pos:
                x, y = pos[node_id]
                x_vals.append(x)
                y_vals.append(y)

                bc = row['betweenness_centrality']
                size = max(8, min(40, bc * 500 + 8))
                sizes.append(size)
                texts.append(node_id.replace('_', ' '))
                hovers.append(
                    f"<b>{node_id}</b><br>"
                    f"Type: {row['node_type']}<br>"
                    f"Betweenness: {bc:.4f}<br>"
                    f"PageRank: {row['pagerank']:.4f}<br>"
                    f"Degree: {row['degree_centrality']:.4f}<br>"
                    f"Risk: {risk_tier}"
                )

        if x_vals:
            node_traces.append(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers+text',
                marker=dict(size=sizes, color=color, line=dict(width=1, color='white')),
                text=texts,
                textposition='top center',
                textfont=dict(size=7, color='white'),
                hovertext=hovers,
                hoverinfo='text',
                name=f'{risk_tier} Risk',
            ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text='Supply Chain Network — Risk-Weighted Graph',
                font=dict(size=20, color='white'),
            ),
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='#0D0D0D',
            paper_bgcolor='#0D0D0D',
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#333',
                font=dict(color='white'),
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )

    return fig


def run_module_a():
    """Execute the full Module A pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Module A — Supply Chain Network Graph               ║")
    print("╚══════════════════════════════════════════════════════╝")

    con = duckdb.connect(DB_PATH, read_only=True)

    # Build graph
    G = build_supply_chain_graph(con)

    # Compute centrality
    centrality_df = compute_centrality_metrics(G)

    # Run simulations
    sim_results = run_historical_simulations(G)

    # Create visualization
    fig = create_network_plotly_figure(G, centrality_df)

    con.close()

    return {
        'graph': G,
        'centrality': centrality_df,
        'simulations': sim_results,
        'figure': fig,
    }


if __name__ == '__main__':
    results = run_module_a()
    print("\n── Centrality Summary ──")
    print(results['centrality'].sort_values('betweenness_centrality', ascending=False).head(15).to_string())
    print("\n── Simulation Results ──")
    print(results['simulations'].to_string())
