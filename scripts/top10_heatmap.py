import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_predictor import EnsemblePredictor

def generate_top10_heatmap(surface="Hard", level="G"):
    logging = joblib.load('feature_state.joblib')
    predictor = EnsemblePredictor()
    
    # 1. Get Top 10 players
    players = []
    for name, stats in logging['player_stats'].items():
        players.append({'name': name, 'rank': stats['rank']})
    
    top_10 = pd.DataFrame(players).sort_values('rank').head(10)['name'].tolist()
    
    # 2. Build Probability Matrix
    n = len(top_10)
    matrix = np.zeros((n, n))
    
    print(f"Calculating {n*n} matchups for {surface} surface...")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.5 # Neutral
                continue
            
            p1, p2 = top_10[i], top_10[j]
            # Use ensemble prediction (weighted avg of XGB and Logistic)
            prob = predictor.predict_match(p1, p2, surface, level)
            matrix[i, j] = prob
            
    # 3. Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="RdYlGn", 
                xticklabels=top_10, yticklabels=top_10, 
                cbar_kws={'label': 'Win Probability (Row beats Column)'})
    
    plt.title(f"ATP Top 10 Matchup Heatmap: {surface} Court ({level} Level)", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top10_heatmap.png')
    print("\nHeatmap saved to plots/top10_heatmap.png")

if __name__ == "__main__":
    generate_top10_heatmap()
