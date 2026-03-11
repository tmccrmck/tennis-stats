import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def generate_handedness_height_serve():
    print("Analyzing Height vs. Serve Efficiency by Handedness...")
    df = load_data()
    pm_df = get_consolidated_player_df(df)
    
    # Filter for known handedness and significant match volume
    pm_df = pm_df[pm_df['hand'].isin(['R', 'L'])]
    
    # Calculate player career stats
    player_stats = pm_df.groupby(['name', 'hand']).agg({
        'ht': 'first',
        'ace': 'sum',
        'svpt': 'sum',
        'win': 'count'
    }).reset_index()
    
    player_stats = player_stats[player_stats['win'] >= 30]
    player_stats['ace_rate'] = (player_stats['ace'] / player_stats['svpt']) * 100
    
    # Visualization
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot with regression lines per hand
    scatter = sns.lmplot(
        data=player_stats, x='ht', y='ace_rate', hue='hand',
        palette={'R': '#1f77b4', 'L': '#ff7f0e'},
        height=8, aspect=1.5,
        scatter_kws={'alpha': 0.5, 's': 80},
        line_kws={'linewidth': 3}
    )
    
    plt.title("The 'Southpaw Leverage': Ace Rate vs. Height by Handedness (2019-2024)", fontsize=20)
    plt.xlabel("Player Height (cm)", fontsize=14)
    plt.ylabel("Aces per 100 Service Points", fontsize=14)
    
    # Annotate significant players
    from adjustText import adjust_text
    ax = plt.gca()
    texts = []
    
    notable = ["John Isner", "Reilly Opelka", "Nick Kyrgios", "Hubert Hurkacz", 
               "Ben Shelton", "Denis Shapovalov", "Cameron Norrie", "Rafael Nadal",
               "Novak Djokovic", "Jannik Sinner", "Carlos Alcaraz", "Daniil Medvedev"]
    
    for name in notable:
        if name in player_stats['name'].values:
            row = player_stats[player_stats['name'] == name].iloc[0]
            texts.append(ax.text(row['ht'], row['ace_rate'], name, fontsize=10, weight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.tight_layout()
    plt.savefig('plots/height_vs_serve_by_hand.png')
    print("Plot saved to plots/height_vs_serve_by_hand.png")

if __name__ == "__main__":
    generate_handedness_height_serve()
