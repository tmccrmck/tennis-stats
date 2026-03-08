import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def generate_compensation_analysis():
    print("Analyzing compensatory skills for shorter players...")
    df = load_data()
    pm_df = get_consolidated_player_df(df)
    
    # Aggregate stats per player
    # We need Return Points Won % and BP Conversion %
    player_stats = pm_df.groupby('name').agg({
        'ht': 'first',
        'win': ['mean', 'count'],
        'opp_svpt': 'sum',
        'opp_1stWon': 'sum',
        'opp_2ndWon': 'sum',
        'opp_bpSaved': 'sum',
        'opp_bpFaced': 'sum'
    })
    
    player_stats.columns = ['ht', 'win_rate', 'match_count', 'opp_svpt', 'opp_1stW', 'opp_2ndW', 'opp_bps', 'opp_bpf']
    
    # Filter for significant sample size
    player_stats = player_stats[player_stats['match_count'] >= 30]
    
    # Calculate Return Mastery (Compensatory Skill)
    # Return Pts Won %
    ret_won = player_stats['opp_svpt'] - (player_stats['opp_1stW'] + player_stats['opp_2ndW'])
    player_stats['ret_won_pct'] = (ret_won / player_stats['opp_svpt']) * 100
    
    # BP Conversion %
    player_stats['bp_conv_pct'] = ((player_stats['opp_bpf'] - player_stats['opp_bps']) / np.maximum(1, player_stats['opp_bpf'])) * 100
    
    # Combined Mastery Index
    player_stats['return_mastery'] = (player_stats['ret_won_pct'] + player_stats['bp_conv_pct']) / 2
    
    # 3. Plot
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot
    scatter = plt.scatter(
        player_stats['ht'], 
        player_stats['return_mastery'],
        c=player_stats['win_rate'], 
        s=player_stats['match_count']*2, 
        cmap='plasma', 
        alpha=0.6, 
        edgecolors='w'
    )
    
    plt.colorbar(scatter, label='Career Win %')
    
    # Elite Threshold Line
    sns.regplot(data=player_stats, x='ht', y='return_mastery', 
                scatter=False, color='red', line_kws={'linestyle': '--', 'alpha': 0.5})
    
    # Annotate MANY players using adjustText
    from adjustText import adjust_text
    texts = []
    
    # Combine top 30 by win rate + our targets
    top_performers = player_stats.sort_values('win_rate', ascending=False).head(30).index.tolist()
    targets = list(set(top_performers + [
        "Diego Schwartzman", "Alex De Minaur", "Kei Nishikori", "Sebastian Baez",
        "Novak Djokovic", "Jannik Sinner", "Carlos Alcaraz", "Rafael Nadal", "Roger Federer",
        "John Isner", "Reilly Opelka", "Daniil Medvedev", "Alexander Zverev", "Nick Kyrgios",
        "Holger Rune", "Taylor Fritz", "Casper Ruud", "Andrey Rublev", "Stefanos Tsitsipas",
        "Hubert Hurkacz", "Frances Tiafoe", "Tommy Paul", "Ben Shelton"
    ]))
    
    for name in targets:
        if name in player_stats.index:
            row = player_stats.loc[name]
            texts.append(plt.text(row['ht'], row['return_mastery'], name, fontsize=9, weight='bold'))

    # Automatically adjust labels to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title("The 'Compensatory Skill' Map: Return Mastery vs. Height", fontsize=22)
    plt.xlabel("Player Height (cm)", fontsize=14)
    plt.ylabel("Return Mastery Index (Return Pts Won % + BP Conv %)", fontsize=14)
    
    # Quadrant Labels
    plt.text(170, player_stats['return_mastery'].max(), "The Scrambler's Requirement:\nElite Return Skills", color='darkred', fontsize=12, style='italic')
    plt.text(205, player_stats['return_mastery'].min(), "The Giant's Luxury:\nLower Return Requirement", color='darkblue', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('plots/short_player_compensation.png')
    print("Compensation analysis saved to plots/short_player_compensation.png")

if __name__ == "__main__":
    generate_compensation_analysis()
