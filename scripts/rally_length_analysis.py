import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data

def generate_rally_height_analysis():
    print("Loading and processing rally data...")
    # 1. Load height mapping
    atp_df = load_data()
    winners_ht = atp_df[['winner_name', 'winner_ht']].rename(columns={'winner_name': 'name', 'winner_ht': 'ht'})
    losers_ht = atp_df[['loser_name', 'loser_ht']].rename(columns={'loser_name': 'name', 'loser_ht': 'ht'})
    player_heights = pd.concat([winners_ht, losers_ht]).drop_duplicates('name').dropna()
    
    # 2. Load MCP Rally data
    rally_df = pd.read_csv('data/charting-m-stats-Rally.csv', low_memory=False)
    
    # We need to map match_id to actual player names
    # Load match metadata
    matches_df = pd.read_csv('data/charting-m-matches.csv')
    matches_df = matches_df[['match_id', 'Player 1', 'Player 2']]
    
    # Merge names onto rally data
    rally_df = rally_df.merge(matches_df, on='match_id')
    
    # Filter for the relevant rows (1-3 and 10+)
    # We'll use the '1-3' and '10' rows (which represent 10+ shots)
    short_pts = rally_df[rally_df['row'] == '1-3'].copy()
    long_pts = rally_df[rally_df['row'] == '10'].copy()
    
    def process_pts(df, label):
        # We need to calculate win rate for Player 1 and Player 2 separately
        p1 = df[['Player 1', 'pl1_won', 'pts']].rename(columns={'Player 1': 'name', 'pl1_won': 'won', 'pts': 'total'})
        p2 = df[['Player 2', 'pl2_won', 'pts']].rename(columns={'Player 2': 'name', 'pl2_won': 'won', 'pts': 'total'})
        combined = pd.concat([p1, p2])
        agg = combined.groupby('name').agg({'won': 'sum', 'total': 'sum'})
        agg[f'{label}_win_pct'] = (agg['won'] / np.maximum(1, agg['total'])) * 100
        return agg[[f'{label}_win_pct']]

    short_stats = process_pts(short_pts, 'short')
    long_stats = process_pts(long_pts, 'long')
    
    # Merge and add heights
    stats = short_stats.join(long_stats, how='inner')
    stats = stats.merge(player_heights, left_index=True, right_on='name')
    
    # Filter for significant data
    # (Since process_pts doesn't give us match count, we'll assume players in both sets are significant)
    stats['rally_impact'] = stats['long_win_pct'] - stats['short_win_pct']
    
    # 3. Visualization
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot Height vs Rally Impact
    # Positive impact = Better at long rallies
    # Negative impact = Better at short points
    sns.regplot(data=stats, x='ht', y='rally_impact', 
                scatter_kws={'alpha':0.5, 'color':'blue'}, line_kws={'color':'red'})
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Annotate Top Players
    notable = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev", 
               "Alexander Zverev", "Taylor Fritz", "Alex De Minaur", "Diego Schwartzman", 
               "John Isner", "Reilly Opelka"]
    
    for name in notable:
        if name in stats['name'].values:
            row = stats[stats['name'] == name].iloc[0]
            plt.annotate(name, (row['ht'], row['rally_impact']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

    plt.title("Rally Length Resilience: Win % Delta (10+ shots vs 1-3 shots) by Height", fontsize=18)
    plt.xlabel("Player Height (cm)", fontsize=14)
    plt.ylabel("Rally Impact (Long Win % - Short Win %)", fontsize=14)
    
    # Explanation
    plt.text(stats['ht'].min(), stats['rally_impact'].max(), "Thrives in Long Rallies", color='darkgreen', fontsize=12, style='italic')
    plt.text(stats['ht'].min(), stats['rally_impact'].min(), "Short Point Specialist", color='darkred', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('plots/rally_length_impact.png')
    print("Rally analysis saved to plots/rally_length_impact.png")

if __name__ == "__main__":
    generate_rally_height_analysis()
