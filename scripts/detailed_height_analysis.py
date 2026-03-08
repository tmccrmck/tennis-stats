import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data

def generate_detailed_height_analysis():
    print("Loading and merging datasets...")
    # 1. Load ATP Match data (for heights)
    atp_df = load_data(years=['2024'])
    
    # Create a simple player -> height mapping
    winners_ht = atp_df[['winner_name', 'winner_ht']].rename(columns={'winner_name': 'name', 'winner_ht': 'ht'})
    losers_ht = atp_df[['loser_name', 'loser_ht']].rename(columns={'loser_name': 'name', 'loser_ht': 'ht'})
    player_heights = pd.concat([winners_ht, losers_ht]).drop_duplicates('name').dropna()
    
    # 2. Load Match Charting Project (MCP) data
    mcp_stats = pd.read_csv('data/charting-m-stats-Overview.csv', low_memory=False)
    # We only want the 'Total' rows (aggregate stats for the match)
    mcp_stats = mcp_stats[mcp_stats['set'] == 'Total']
    
    # 3. Merge MCP stats with heights
    # The MCP uses names like 'Novak Djokovic', matching our ATP names
    merged = mcp_stats.merge(player_heights, left_on='player', right_on='name')
    
    # 4. Calculate Rates
    # Total points played = serve_pts + return_pts
    merged['total_pts'] = merged['serve_pts'] + merged['return_pts']
    merged = merged[merged['total_pts'] > 50] # Filter out short/incomplete matches
    
    merged['winner_rate'] = (merged['winners'] / merged['total_pts']) * 100
    merged['error_rate'] = (merged['unforced'] / merged['total_pts']) * 100
    
    # Group by player to get average rates
    player_analysis = merged.groupby('player').agg({
        'ht': 'first',
        'winner_rate': 'mean',
        'error_rate': 'mean',
        'total_pts': 'count'
    }).rename(columns={'total_pts': 'match_count'})
    
    # Filter for significant sample size
    player_analysis = player_analysis[player_analysis['match_count'] >= 3]
    
    # 5. Plotting
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # We'll use two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('2024 Detailed Stats vs Height (Match Charting Project Data)', fontsize=20)
    
    # Plot 1: Height vs Winner Rate
    sns.regplot(ax=ax1, data=player_analysis, x='ht', y='winner_rate', 
                scatter_kws={'alpha':0.5, 'color':'green'}, line_kws={'color':'red'})
    ax1.set_title('Aggression: Winner Rate % vs Height')
    ax1.set_xlabel('Height (cm)')
    ax1.set_ylabel('Winners per 100 Points')
    
    # Plot 2: Height vs Error Rate
    sns.regplot(ax=ax2, data=player_analysis, x='ht', y='error_rate', 
                scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'red'})
    ax2.set_title('Consistency: Unforced Error Rate % vs Height')
    ax2.set_xlabel('Height (cm)')
    ax2.set_ylabel('Unforced Errors per 100 Points')
    
    # Annotate Top Players
    notable = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev", "Alexander Zverev", "Taylor Fritz", "Alex De Minaur"]
    for name in notable:
        if name in player_analysis.index:
            ax1.annotate(name, (player_analysis.loc[name, 'ht'], player_analysis.loc[name, 'winner_rate']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
            ax2.annotate(name, (player_analysis.loc[name, 'ht'], player_analysis.loc[name, 'error_rate']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/detailed_height_analysis.png')
    print("Detailed height analysis saved to plots/detailed_height_analysis.png")

if __name__ == "__main__":
    generate_detailed_height_analysis()
