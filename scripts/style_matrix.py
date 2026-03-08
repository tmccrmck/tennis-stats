import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df, plot_pca_map

def generate_style_matrix():
    print("Extracting stylistic features for PCA...")
    df = load_data()
    all_player_matches = get_consolidated_player_df(df)
    
    # 1. Aggregate Stylistic Metrics
    player_stats = all_player_matches.groupby('name').agg({
        'ht': 'first',
        'ace': 'mean',
        'win': ['mean', 'count'],
        'SvGms': 'sum',
        'bpSaved': 'sum',
        'bpFaced': 'sum',
        'opp_SvGms': 'sum',
        'opp_bpSaved': 'sum',
        'opp_bpFaced': 'sum'
    })
    
    player_stats.columns = ['ht', 'ace_avg', 'win_rate', 'match_count', 
                            'total_svgms', 'total_bps', 'total_bpf', 
                            'total_opp_svgms', 'total_opp_bps', 'total_opp_bpf']
    
    player_stats = player_stats[player_stats['match_count'] >= 30]
    
    # Ratios
    player_stats['hold_pct'] = (player_stats['total_svgms'] - (player_stats['total_bpf'] - player_stats['total_bps'])) / player_stats['total_svgms']
    player_stats['break_pct'] = (player_stats['total_opp_bpf'] - player_stats['total_opp_bps']) / player_stats['total_opp_svgms']
    player_stats['bp_saved_pct'] = player_stats['total_bps'] / np.maximum(1, player_stats['total_bpf'])
    player_stats['bp_conv_pct'] = (player_stats['total_opp_bpf'] - player_stats['total_opp_bps']) / np.maximum(1, player_stats['total_opp_bpf'])
    
    # 2. PCA
    style_cols = ['ace_avg', 'hold_pct', 'break_pct', 'bp_saved_pct', 'bp_conv_pct']
    X = player_stats[style_cols]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    player_stats['PC1'] = X_pca[:, 0]
    player_stats['PC2'] = X_pca[:, 1]
    
    # 3. Plot
    plot_pca_map(
        player_stats, 
        title="ATP Style Matrix: PCA Mapping of Player DNA",
        xlabel="Principal Component 1 (Service Power <---> Return Skill)",
        ylabel="Principal Component 2 (Consistency & Clutch Factor)",
        filename="style_matrix_pca.png",
        hue_col='ht'
    )

if __name__ == "__main__":
    generate_style_matrix()
