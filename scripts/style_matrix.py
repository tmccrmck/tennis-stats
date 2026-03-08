import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def generate_style_matrix():
    print("Extracting stylistic features for PCA...")
    df = load_data()
    all_player_matches = get_consolidated_player_df(df)
    
    # 1. Aggregate Stylistic Metrics
    player_stats = all_player_matches.groupby('name').agg({
        'ht': 'first',
        'ace': 'mean',
        'df': 'mean',
        'win': ['mean', 'count'],
        'SvGms': 'sum',
        'bpSaved': 'sum',
        'bpFaced': 'sum',
        'opp_SvGms': 'sum',
        'opp_bpSaved': 'sum',
        'opp_bpFaced': 'sum'
    })
    
    # Flatten columns
    player_stats.columns = ['ht', 'ace_avg', 'df_avg', 'win_rate', 'match_count', 
                            'total_svgms', 'total_bps', 'total_bpf', 
                            'total_opp_svgms', 'total_opp_bps', 'total_opp_bpf']
    
    # Filter for significant sample size (min 30 matches)
    player_stats = player_stats[player_stats['match_count'] >= 30]
    
    # Calculate Ratios
    player_stats['hold_pct'] = (player_stats['total_svgms'] - (player_stats['total_bpf'] - player_stats['total_bps'])) / player_stats['total_svgms']
    player_stats['break_pct'] = (player_stats['total_opp_bpf'] - player_stats['total_opp_bps']) / player_stats['total_opp_svgms']
    player_stats['bp_saved_pct'] = player_stats['total_bps'] / np.maximum(1, player_stats['total_bpf'])
    player_stats['bp_conv_pct'] = (player_stats['total_opp_bpf'] - player_stats['total_opp_bps']) / np.maximum(1, player_stats['total_opp_bpf'])
    
    # 2. Prepare for PCA
    style_cols = ['ace_avg', 'hold_pct', 'break_pct', 'bp_saved_pct', 'bp_conv_pct']
    X = player_stats[style_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    player_stats['PC1'] = X_pca[:, 0]
    player_stats['PC2'] = X_pca[:, 1]
    
    # 3. Plot Style Matrix
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="white")
    
    # Scatter plot colored by height (to see physical correlation)
    scatter = plt.scatter(player_stats['PC1'], player_stats['PC2'], 
                         c=player_stats['ht'], s=100, 
                         cmap='coolwarm', alpha=0.7, edgecolors='gray')
    
    plt.colorbar(scatter, label='Player Height (cm)')
    
    # Annotate Top Players and Style Extremes
    targets = [
        "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev",
        "Alexander Zverev", "Taylor Fritz", "Alex De Minaur", "Casper Ruud",
        "Nick Kyrgios", "John Isner", "Reilly Opelka", "Diego Schwartzman",
        "Rafael Nadal", "Roger Federer", "Holger Rune"
    ]
    
    for name in targets:
        if name in player_stats.index:
            plt.annotate(name, (player_stats.loc[name, 'PC1'], player_stats.loc[name, 'PC2']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=11, weight='bold')

    # Explain the Axes (based on PCA loadings)
    # Component 1 usually separates Returners from Servers
    # Component 2 usually captures overall Efficiency/Clutch
    plt.title("ATP Style Matrix: PCA Mapping of Player DNA (2019-2024)", fontsize=22)
    plt.xlabel("Principal Component 1 (Service Power <---> Return Skill)", fontsize=14)
    plt.ylabel("Principal Component 2 (Consistency & Clutch Factor)", fontsize=14)
    
    # Add quadrant labels
    plt.text(player_stats['PC1'].min(), player_stats['PC2'].max(), "Elite Scramblers", fontsize=12, alpha=0.5, style='italic')
    plt.text(player_stats['PC1'].max(), player_stats['PC2'].max(), "Dominant Servers", fontsize=12, alpha=0.5, style='italic')
    
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    
    plt.tight_layout()
    plt.savefig('plots/style_matrix_pca.png')
    print("Matrix saved to plots/style_matrix_pca.png")

if __name__ == "__main__":
    generate_style_matrix()
