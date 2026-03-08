import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_mcp_data, DEFAULT_TARGETS

def generate_efficiency_frontier():
    print("Calculating Winner/Error efficiency rates...")
    # Load MCP Overview stats
    mcp_df = pd.read_csv('data/charting-m-stats-Overview.csv', low_memory=False)
    mcp_df = mcp_df[mcp_df['set'] == 'Total']
    
    # Calculate rates per match
    mcp_df['total_pts'] = mcp_df['serve_pts'] + mcp_df['return_pts']
    mcp_df = mcp_df[mcp_df['total_pts'] > 50]
    
    mcp_df['winner_rate'] = (mcp_df['winners'] / mcp_df['total_pts']) * 100
    mcp_df['error_rate'] = (mcp_df['unforced'] / mcp_df['total_pts']) * 100
    
    # Aggregate by player
    player_stats = mcp_df.groupby('player').agg({
        'winner_rate': 'mean',
        'error_rate': 'mean',
        'match_id': 'count'
    }).rename(columns={'match_id': 'match_count'})
    
    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 5]
    
    # 3. Plot
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot colored by winner rate
    scatter = plt.scatter(
        player_stats['error_rate'], 
        player_stats['winner_rate'],
        c=player_stats['winner_rate'] / np.maximum(1, player_stats['error_rate']), 
        s=100, 
        cmap='RdYlGn', 
        alpha=0.6, 
        edgecolors='gray'
    )
    
    plt.colorbar(scatter, label='Winner/Error Ratio')
    
    # Annotate Top Players
    for name in DEFAULT_TARGETS:
        if name in player_stats.index:
            plt.annotate(name, (player_stats.loc[name, 'error_rate'], player_stats.loc[name, 'winner_rate']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

    plt.title("The ATP Efficiency Frontier: Winner Rate vs. Unforced Error Rate (2019-2024)", fontsize=20)
    plt.xlabel("Unforced Errors per 100 Points (Lower is Better)", fontsize=14)
    plt.ylabel("Winners per 100 Points (Higher is Better)", fontsize=14)
    
    # Flip X axis so "Better" (low error) is on the right? 
    # Actually, keep it standard but add labels
    plt.text(player_stats['error_rate'].min(), player_stats['winner_rate'].max(), "Elite Efficiency", color='darkgreen', fontsize=12, weight='bold')
    plt.text(player_stats['error_rate'].max()-2, player_stats['winner_rate'].min()+1, "High Variance / Risk", color='darkred', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('plots/efficiency_frontier.png')
    print("Efficiency Frontier plot saved to plots/efficiency_frontier.png")

if __name__ == "__main__":
    generate_efficiency_frontier()
