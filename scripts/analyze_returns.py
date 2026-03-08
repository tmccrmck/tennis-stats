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
        'opp_svpt': 'sum',
        'opp_1stWon': 'sum',
        'opp_2ndWon': 'sum',
        'opp_bpFaced': 'sum',
        'opp_bpSaved': 'sum',
        'name': 'count'
    }).rename(columns={'name': 'match_count'})

    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 5]

    # Calculate return percentages
    # Return Pts Won = Opponent Service Points - Opponent Service Points Won
    ret_pts_won = player_stats['opp_svpt'] - (player_stats['opp_1stWon'] + player_stats['opp_2ndWon'])
    player_stats['ret_pts_won_pct'] = (ret_pts_won / player_stats['opp_svpt']) * 100
    
    # BP Converted = Opponent Break Points Faced - Opponent Break Points Saved
    bp_converted = player_stats['opp_bpFaced'] - player_stats['opp_bpSaved']
    player_stats['bp_conv_pct'] = (bp_converted / player_stats['opp_bpFaced']) * 100

    # 3. Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('ATP 2020-2024: Return Stats vs Player Height (Min 5 matches)', fontsize=20)

    # 1. Return Points Won % vs Height
    sns.regplot(ax=axes[0], data=player_stats, x='ht', y='ret_pts_won_pct', scatter_kws={'alpha':0.5, 'color':'teal'}, line_kws={'color':'red'})
    axes[0].set_title('Return Points Won % vs Height')
    axes[0].set_xlabel('Height (cm)')
    axes[0].set_ylabel('Return Points Won %')

    # 2. Break Point Conversion % vs Height
    sns.regplot(ax=axes[1], data=player_stats, x='ht', y='bp_conv_pct', scatter_kws={'alpha':0.5, 'color':'coral'}, line_kws={'color':'red'})
    axes[1].set_title('Break Point Conversion % vs Height')
    axes[1].set_xlabel('Height (cm)')
    axes[1].set_ylabel('BP Conversion %')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/height_vs_return_stats.png')
    print("Visualization saved to plots/height_vs_return_stats.png")

if __name__ == "__main__":
    run_analysis()
