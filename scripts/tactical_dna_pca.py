import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_mcp_data, plot_pca_map

def generate_tactical_dna():
    print("Processing tactical direction data...")
    dir_df = load_mcp_data('charting-m-stats-ShotDirection.csv')
    
    # Filter for Forehand (F) and Backhand (B) rows
    f_shots = dir_df[dir_df['row'] == 'F'].copy()
    b_shots = dir_df[dir_df['row'] == 'B'].copy()
    
    def calculate_ratios(df, prefix):
        agg = df.groupby('player').agg({
            'crosscourt': 'sum', 'down_middle': 'sum', 'down_the_line': 'sum',
            'inside_out': 'sum', 'inside_in': 'sum', 'match_id': 'count'
        })
        # Normalize
        cols = ['crosscourt', 'down_middle', 'down_the_line', 'inside_out', 'inside_in']
        total = agg[cols].sum(axis=1)
        agg[cols] = agg[cols].div(total, axis=0) * 100
        # Indicators
        agg[f'{prefix}_aggression'] = agg['down_the_line'] + agg['inside_in']
        agg[f'{prefix}_cross'] = agg['crosscourt']
        return agg[agg['match_id'] >= 5][[f'{prefix}_aggression', f'{prefix}_cross']]

    f_ratios = calculate_ratios(f_shots, 'FH')
    b_ratios = calculate_ratios(b_shots, 'BH')
    tactical_stats = f_ratios.join(b_ratios, how='inner')
    
    # PCA
    features = ['FH_aggression', 'FH_cross', 'BH_aggression', 'BH_cross']
    X_scaled = StandardScaler().fit_transform(tactical_stats[features])
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    tactical_stats['PC1'] = X_pca[:, 0]
    tactical_stats['PC2'] = X_pca[:, 1]
    
    # Plot
    plot_pca_map(
        tactical_stats,
        title="ATP Tactical DNA: Shot Placement Preference PCA",
        xlabel="Principal Component 1 (Conservative Patterns <---> Aggressive Changes)",
        ylabel="Principal Component 2 (Forehand vs Backhand Dominance)",
        filename="tactical_dna_pca.png"
    )

if __name__ == "__main__":
    generate_tactical_dna()
