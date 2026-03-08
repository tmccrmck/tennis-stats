import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os
import re

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data

def count_tiebreaks(score_str):
    """Counts number of tiebreaks in a match score string."""
    if pd.isna(score_str): return 0
    # Search for (7-6) or 7-6 pattern
    return len(re.findall(r'7-6|6-7', str(score_str)))

def generate_tournament_flavor():
    print("Extracting tournament characteristics...")
    df = load_data()
    
    # Filter for matches with stats
    df = df.dropna(subset=['w_ace', 'l_ace', 'minutes', 'w_bpFaced', 'l_bpFaced', 'score'])
    
    # 1. Feature Engineering per Match
    df['total_aces'] = df['w_ace'] + df['l_ace']
    df['total_bp'] = df['w_bpFaced'] + df['l_bpFaced']
    df['tiebreaks'] = df['score'].apply(count_tiebreaks)
    
    # 2. Group by Tournament
    # We use tourney_name to group multiple years of the same event
    tourney_stats = df.groupby(['tourney_name']).agg({
        'surface': 'first',
        'total_aces': 'mean',
        'minutes': 'mean',
        'total_bp': 'mean',
        'tiebreaks': 'mean',
        'match_num': 'count'
    }).rename(columns={
        'total_aces': 'avg_aces',
        'minutes': 'avg_minutes',
        'total_bp': 'avg_bp',
        'tiebreaks': 'avg_tb',
        'match_num': 'match_count'
    })
    
    # Filter for tournaments with significant data
    tourney_stats = tourney_stats[tourney_stats['match_count'] >= 100]
    
    # 3. PCA
    features = ['avg_aces', 'avg_minutes', 'avg_bp', 'avg_tb']
    X = tourney_stats[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    tourney_stats['PC1'] = X_pca[:, 0]
    tourney_stats['PC2'] = X_pca[:, 1]
    
    # 4. Visualization
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Color by surface to see how well it explains the variance
    scatter = sns.scatterplot(
        data=tourney_stats, x='PC1', y='PC2', 
        hue='surface', style='surface',
        s=150, alpha=0.8, palette='Set1'
    )
    
    # Annotate significant tournaments
    notable = [
        "Wimbledon", "Roland Garros", "US Open", "Australian Open",
        "Indian Wells Masters", "Miami Masters", "Monte Carlo Masters",
        "Madrid Masters", "Rome Masters", "Cincinnati Masters",
        "Paris Masters", "Shanghai Masters", "Canada Masters"
    ]
    
    for name in notable:
        if name in tourney_stats.index:
            plt.annotate(name, (tourney_stats.loc[name, 'PC1'], tourney_stats.loc[name, 'PC2']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

    plt.title("Tournament Flavor PCA: Clustering ATP Events by Match DNA (2019-2024)", fontsize=18)
    plt.xlabel("Principal Component 1 (Service Dominance <---> Long Rallies/Grind)", fontsize=12)
    plt.ylabel("Principal Component 2 (Volatility & Break Potential)", fontsize=12)
    
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    
    plt.tight_layout()
    plt.savefig('plots/tournament_flavor_pca.png')
    print("Tournament PCA saved to plots/tournament_flavor_pca.png")

if __name__ == "__main__":
    generate_tournament_flavor()
