import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df, DEFAULT_TARGETS

def generate_style_umap():
    print("Extracting features for UMAP projection...")
    df = load_data()
    all_player_matches = get_consolidated_player_df(df)
    
    # 1. Aggregate Stylistic Metrics (Deep Profile)
    player_stats = all_player_matches.groupby('name').agg({
        'ht': 'first',
        'ace': 'mean',
        'df': 'mean',
        'win': ['mean', 'count'],
        'SvGms': 'sum',
        'bpSaved': 'sum',
        'bpFaced': 'sum',
        'opp_svpt': 'sum',
        'opp_1stWon': 'sum',
        'opp_2ndWon': 'sum',
        'opp_bpSaved': 'sum',
        'opp_bpFaced': 'sum'
    })
    
    player_stats.columns = [
        'ht', 'ace_avg', 'df_avg', 'win_rate', 'match_count', 
        'total_svgms', 'total_bps', 'total_bpf', 
        'opp_svpt', 'opp_1stW', 'opp_2ndW', 'total_opp_bps', 'total_opp_bpf'
    ]
    
    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 40]
    
    # Ratios
    player_stats['hold_pct'] = (player_stats['total_svgms'] - (player_stats['total_bpf'] - player_stats['total_bps'])) / player_stats['total_svgms']
    ret_won = player_stats['opp_svpt'] - (player_stats['opp_1stW'] + player_stats['opp_2ndW'])
    player_stats['ret_won_pct'] = (ret_won / player_stats['opp_svpt']) * 100
    player_stats['bp_saved_pct'] = player_stats['total_bps'] / np.maximum(1, player_stats['total_bpf'])
    player_stats['bp_conv_pct'] = (player_stats['total_opp_bpf'] - player_stats['total_opp_bps']) / np.maximum(1, player_stats['total_opp_bpf'])
    
    # 2. UMAP
    # We use a broader set of features for UMAP to find deeper clusters
    features = ['ht', 'ace_avg', 'df_avg', 'win_rate', 'hold_pct', 'ret_won_pct', 'bp_saved_pct', 'bp_conv_pct']
    X = player_stats[features]
    X_scaled = StandardScaler().fit_transform(X)
    
    print("Running UMAP (n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    player_stats['UMAP1'] = embedding[:, 0]
    player_stats['UMAP2'] = embedding[:, 1]
    
    # 3. Visualization
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="white")
    
    # Color by height to see if UMAP finds physical clusters
    scatter = plt.scatter(
        player_stats['UMAP1'], player_stats['UMAP2'],
        c=player_stats['ht'], cmap='coolwarm',
        s=100, alpha=0.7, edgecolors='gray'
    )
    
    plt.colorbar(scatter, label='Height (cm)')
    
    # Annotate using adjustText
    from adjustText import adjust_text
    texts = []
    
    # Focus on the Top 40 by win rate for clarity in the UMAP space
    top_players = player_stats.sort_values('win_rate', ascending=False).head(40).index.tolist()
    targets = list(set(top_players + DEFAULT_TARGETS))
    
    for name in targets:
        if name in player_stats.index:
            row = player_stats.loc[name]
            texts.append(plt.text(row['UMAP1'], row['UMAP2'], name, fontsize=9, weight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title("ATP Stylistic Manifold: UMAP Projection of Player DNA (2019-2024)", fontsize=22)
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/style_umap.png')
    print("UMAP style map saved to plots/style_umap.png")

if __name__ == "__main__":
    generate_style_umap()
