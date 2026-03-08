import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def run_analysis():
    # 1. Load and Consolidate
    df = load_data()
    all_matches = get_consolidated_player_df(df)

    # 2. Aggregate by player
    player_stats = all_matches.groupby('name').agg({
        'ht': 'first',
        'ace': 'mean',
        'df': 'mean',
        'svpt': 'sum',
        '1stIn': 'sum',
        '1stWon': 'sum',
        '2ndWon': 'sum',
        'name': 'count'
    }).rename(columns={'name': 'match_count'})

    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 5]

    # Calculate percentages
    player_stats['first_serve_win_pct'] = (player_stats['1stWon'] / player_stats['1stIn']) * 100
    player_stats['service_points_won_pct'] = ((player_stats['1stWon'] + player_stats['2ndWon']) / player_stats['svpt']) * 100

    # 3. Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ATP 2020-2024: Serve Stats vs Player Height (Min 5 matches)', fontsize=20)

    plots = [
        (axes[0, 0], 'ace', 'Avg Aces per Match', 'Avg Aces', 'blue'),
        (axes[0, 1], 'first_serve_win_pct', '1st Serve Win %', '1st Serve Win %', 'green'),
        (axes[1, 0], 'df', 'Avg Double Faults per Match', 'Avg Double Faults', 'purple'),
        (axes[1, 1], 'service_points_won_pct', 'Total Service Points Won %', 'Service Points Won %', 'orange')
    ]

    for ax, col, title, ylabel, color in plots:
        sns.regplot(ax=ax, data=player_stats, x='ht', y=col, scatter_kws={'alpha':0.5, 'color':color}, line_kws={'color':'red'})
        ax.set_title(title)
        ax.set_xlabel('Height (cm)')
        ax.set_ylabel(ylabel)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/height_vs_serve_stats.png')
    print("Visualization saved to plots/height_vs_serve_stats.png")

if __name__ == "__main__":
    run_analysis()
