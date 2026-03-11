import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, load_mcp_data, DEFAULT_TARGETS

def generate_movement_analysis():
    print("Calculating Movement Efficiency Index...")
    
    # 1. Load Match Charting Project Data (Overview for Forced Errors)
    # MCP Overview has 'pl1_forced' and 'pl2_forced' in the Rally file
    rally_df = load_mcp_data('charting-m-stats-Rally.csv')
    
    # Extract opponent forced errors per player
    # A forced error for player 2 is a "defensive win" for player 1
    p1 = rally_df[rally_df['row'] == 'Total'][['Player 1', 'pl2_forced', 'pts']].rename(columns={'Player 1': 'name', 'pl2_forced': 'forced_induced', 'pts': 'total'})
    p2 = rally_df[rally_df['row'] == 'Total'][['Player 2', 'pl1_forced', 'pts']].rename(columns={'Player 2': 'name', 'pl1_forced': 'forced_induced', 'pts': 'total'})
    forced_stats = pd.concat([p1, p2]).groupby('name').agg({'forced_induced': 'sum', 'total': 'sum'})
    forced_stats['forced_rate'] = (forced_stats['forced_induced'] / forced_stats['total']) * 100
    
    # 2. Get Long Rally Win % (10+ shots)
    long_pts = rally_df[rally_df['row'] == '10']
    lp1 = long_pts[['Player 1', 'pl1_won', 'pts']].rename(columns={'Player 1': 'name', 'pl1_won': 'won', 'pts': 'total'})
    lp2 = long_pts[['Player 2', 'pl2_won', 'pts']].rename(columns={'Player 2': 'name', 'pl2_won': 'won', 'pts': 'total'})
    long_stats = pd.concat([lp1, lp2]).groupby('name').agg({'won': 'sum', 'total': 'sum'})
    long_stats['long_win_rate'] = (long_stats['won'] / long_stats['total']) * 100
    
    # 3. Combine and Normalize
    movement_df = forced_stats[['forced_rate']].join(long_stats[['long_win_rate']], how='inner')
    
    # Filter for sample size
    movement_df = movement_df[forced_stats['total'] > 500] # Min points played
    
    # Normalize components to 0-100
    def normalize(s): return 100 * (s - s.min()) / (s.max() - s.min())
    movement_df['score'] = (normalize(movement_df['forced_rate']) + normalize(movement_df['long_win_rate'])) / 2
    
    # 4. Add Heights
    atp_df = load_data()
    winners_ht = atp_df[['winner_name', 'winner_ht']].rename(columns={'winner_name': 'name', 'winner_ht': 'ht'})
    losers_ht = atp_df[['loser_name', 'loser_ht']].rename(columns={'loser_name': 'name', 'loser_ht': 'ht'})
    player_heights = pd.concat([winners_ht, losers_ht]).drop_duplicates('name').dropna()
    
    movement_df = movement_df.merge(player_heights, left_index=True, right_on='name')
    
    # 5. Visualization
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot Height vs Movement Score
    scatter = plt.scatter(
        movement_df['ht'], 
        movement_df['score'],
        c=movement_df['score'], 
        s=100, 
        cmap='winter', 
        alpha=0.6, 
        edgecolors='gray'
    )
    
    plt.colorbar(scatter, label='Movement Score (0-100)')
    
    # Trend line
    sns.regplot(data=movement_df, x='ht', y='score', scatter=False, color='red', line_kws={'linestyle': '--'})
    
    # Annotate Top Movers
    from adjustText import adjust_text
    texts = []
    
    # Custom targets for movement
    movers = ["Alex De Minaur", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev", 
              "Diego Schwartzman", "Rafael Nadal", "Jannik Sinner", "Alexander Zverev",
              "John Isner", "Reilly Opelka", "Sebastian Baez", "Casper Ruud", "Taylor Fritz"]
    
    for name in movers:
        if name in movement_df['name'].values:
            row = movement_df[movement_df['name'] == name].iloc[0]
            texts.append(plt.text(row['ht'], row['score'], name, fontsize=10, weight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title("The 'Mobility Cliff': Movement Efficiency Score vs. Height", fontsize=20)
    plt.xlabel("Player Height (cm)", fontsize=14)
    plt.ylabel("Movement Score (Induced Forced Errors + Long Rally Win %)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/movement_vs_height.png')
    print("Movement analysis saved to plots/movement_vs_height.png")

if __name__ == "__main__":
    generate_movement_analysis()
