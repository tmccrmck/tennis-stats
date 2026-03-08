import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os
import re

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, plot_pca_map

def count_tiebreaks(score_str):
    if pd.isna(score_str): return 0
    return len(re.findall(r'7-6|6-7', str(score_str)))

def generate_tournament_flavor():
    print("Extracting tournament characteristics...")
    df = load_data()
    df = df.dropna(subset=['w_ace', 'l_ace', 'minutes', 'w_bpFaced', 'l_bpFaced', 'score'])
    
    # Feature Engineering
    df['total_aces'] = df['w_ace'] + df['l_ace']
    df['total_bp'] = df['w_bpFaced'] + df['l_bpFaced']
    df['tiebreaks'] = df['score'].apply(count_tiebreaks)
    
    tourney_stats = df.groupby(['tourney_name']).agg({
        'surface': 'first',
        'total_aces': 'mean',
        'minutes': 'mean',
        'total_bp': 'mean',
        'tiebreaks': 'mean',
        'match_num': 'count'
    }).rename(columns={
        'total_aces': 'avg_aces', 'minutes': 'avg_minutes',
        'total_bp': 'avg_bp', 'tiebreaks': 'avg_tb', 'match_num': 'match_count'
    })
    
    tourney_stats = tourney_stats[tourney_stats['match_count'] >= 100]
    
    # PCA
    features = ['avg_aces', 'avg_minutes', 'avg_bp', 'avg_tb']
    X = tourney_stats[features]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    tourney_stats['PC1'] = X_pca[:, 0]
    tourney_stats['PC2'] = X_pca[:, 1]
    
    # Plot
    plot_pca_map(
        tourney_stats,
        title="Tournament Flavor PCA: Clustering ATP Events",
        xlabel="Principal Component 1 (Service Dominance <---> Long Rallies/Grind)",
        ylabel="Principal Component 2 (Volatility & Break Potential)",
        filename="tournament_flavor_pca.png",
        hue_col='surface'
    )

if __name__ == "__main__":
    generate_tournament_flavor()
