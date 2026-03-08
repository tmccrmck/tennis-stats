import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def generate_iceman_index():
    print("Calculating Break Point Efficiency stats...")
    df = load_data()
    all_player_matches = get_consolidated_player_df(df)
    
    # Aggregate stats per player
    player_stats = all_player_matches.groupby('name').agg({
        'bpSaved': 'sum',
        'bpFaced': 'sum',
        'opp_bpSaved': 'sum',
        'opp_bpFaced': 'sum',
        'win': 'count'
    }).rename(columns={'win': 'match_count'})
    
    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 50]
    
    # Calculate percentages
    player_stats['bp_saved_pct'] = (player_stats['bpSaved'] / np.maximum(1, player_stats['bpFaced'])) * 100
    # BP Converted = (Opponent BP Faced - Opponent BP Saved) / Opponent BP Faced
    player_stats['bp_conv_pct'] = ((player_stats['opp_bpFaced'] - player_stats['opp_bpSaved']) / np.maximum(1, player_stats['opp_bpFaced'])) * 100
    
    # 3. Plot
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot colored by match count
    scatter = plt.scatter(
        player_stats['bp_conv_pct'], 
        player_stats['bp_saved_pct'],
        c=player_stats['match_count'], 
        s=100, 
        cmap='viridis', 
        alpha=0.6, 
        edgecolors='w'
    )
    
    plt.colorbar(scatter, label='Matches Played')
    
    # Mean lines for quadrants
    mean_saved = player_stats['bp_saved_pct'].mean()
    mean_conv = player_stats['bp_conv_pct'].mean()
    
    plt.axhline(mean_saved, color='red', linestyle='--', alpha=0.5)
    plt.axvline(mean_conv, color='red', linestyle='--', alpha=0.5)
    
    # Annotate Top Players
    notable = [
        "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev",
        "Alexander Zverev", "Rafael Nadal", "Roger Federer", "Taylor Fritz",
        "Alex De Minaur", "Nick Kyrgios", "Hubert Hurkacz", "Casper Ruud", "Andrey Rublev"
    ]
    
    for name in notable:
        if name in player_stats.index:
            plt.annotate(name, (player_stats.loc[name, 'bp_conv_pct'], player_stats.loc[name, 'bp_saved_pct']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

    # Quadrant Labels
    plt.text(player_stats['bp_conv_pct'].max()-2, player_stats['bp_saved_pct'].max()-1, "The Icemen (Clutch)", fontsize=12, weight='bold', color='darkgreen')
    plt.text(player_stats['bp_conv_pct'].min()+1, player_stats['bp_saved_pct'].min()+1, "Underperformers", fontsize=12, weight='bold', color='darkred')

    plt.title("The 'Iceman' Index: Break Point Efficiency (2019-2024)", fontsize=20)
    plt.xlabel("Break Points Converted % (Offensive Clutch)", fontsize=14)
    plt.ylabel("Break Points Saved % (Defensive Clutch)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/iceman_index.png')
    print("Iceman Index plot saved to plots/iceman_index.png")

if __name__ == "__main__":
    generate_iceman_index()
