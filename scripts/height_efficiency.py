import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def analyze_height_efficiency():
    # 1. Load and Consolidate
    df = load_data()
    all_player_matches = get_consolidated_player_df(df)
    
    # 2. Group by player
    stats = all_player_matches.groupby('name').agg({
        'ht': 'first',
        'win': ['mean', 'count'],
        'SvGms': 'sum',
        'bpFaced': 'sum',
        'bpSaved': 'sum',
        'opp_bpFaced': 'sum',
        'opp_bpSaved': 'sum',
        'opp_SvGms': 'sum'
    })
    
    stats.columns = ['ht', 'win_rate', 'match_count', 'total_svgms', 'total_bpf', 'total_bps', 'total_opp_bpf', 'total_opp_bps', 'total_opp_svgms']
    
    # Filter for significant sample size
    stats = stats[stats['match_count'] >= 20]
    
    # Calculate efficiencies
    stats['hold_pct'] = (stats['total_svgms'] - (stats['total_bpf'] - stats['total_bps'])) / stats['total_svgms']
    stats['break_pct'] = (stats['total_opp_bpf'] - stats['total_opp_bps']) / stats['total_opp_svgms']
    
    # --- Plot 1: The Sweet Spot ---
    plt.figure(figsize=(12, 7))
    sns.regplot(data=stats, x='ht', y='win_rate', order=2, scatter_kws={'alpha':0.4, 'color':'blue'}, line_kws={'color':'red'})
    plt.title(f"The Height 'Sweet Spot': Win % vs Player Height (2020-2024)", fontsize=16)
    plt.xlabel("Height (cm)", fontsize=12)
    plt.ylabel("Overall Win Percentage", fontsize=12)
    plt.savefig('plots/height_winrate_sweetspot.png')
    print("Saved plots/height_winrate_sweetspot.png")
    
    # --- Plot 2: Efficiency Scatter ---
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(stats['hold_pct'], stats['break_pct'], 
                         c=stats['ht'], s=stats['match_count']*5, 
                         cmap='viridis', alpha=0.6, edgecolors='w')
    
    plt.colorbar(scatter, label='Player Height (cm)')
    plt.title("Style Analysis: Service vs Return Efficiency (Bubble Size = Matches Played)", fontsize=16)
    plt.xlabel("Hold Percentage (Service Efficiency)", fontsize=12)
    plt.ylabel("Break Percentage (Return Efficiency)", fontsize=12)
    
    top_names = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev", "Taylor Fritz"]
    for name in top_names:
        if name in stats.index:
            plt.annotate(name, (stats.loc[name, 'hold_pct'], stats.loc[name, 'break_pct']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')
            
    plt.savefig('plots/serve_return_efficiency.png')
    print("Saved plots/serve_return_efficiency.png")

if __name__ == "__main__":
    analyze_height_efficiency()
