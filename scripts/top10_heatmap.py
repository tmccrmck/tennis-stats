import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Add root to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction import Predictor
import config

def generate_top10_heatmap(surface="Hard", level="G"):
    predictor = Predictor()
    
    # 1. Get Top 10 players
    players = []
    for name, stats in predictor.state['player_stats'].items():
        players.append({'name': name, 'rank': stats['rank']})
    
    top_10 = pd.DataFrame(players).sort_values('rank').head(10)['name'].tolist()
    
    # 2. Build Probability Matrix
    n = len(top_10)
    matrix = np.zeros((n, n))
    
    print(f"Calculating {n*n} matchups for {surface} surface...")
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.5
                continue
            matrix[i, j] = predictor.predict(top_10[i], top_10[j], surface, level)
            
    # 3. Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="RdYlGn", 
                xticklabels=top_10, yticklabels=top_10, 
                cbar_kws={'label': 'Win Probability (Row beats Column)'})
    
    plt.title(f"ATP Top 10 Matchup Heatmap: {surface} Court ({level} Level)", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top10_heatmap.png')
    print(f"\nHeatmap saved to plots/top10_heatmap.png")

if __name__ == "__main__":
    generate_top10_heatmap()
