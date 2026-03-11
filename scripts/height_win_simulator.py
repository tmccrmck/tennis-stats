import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import Predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction import Predictor

def simulate_height_influence():
    print("Initializing predictor for height simulation...")
    predictor = Predictor()
    
    # Range of heights to simulate
    heights = np.arange(170, 216, 2)
    surfaces = ['Hard', 'Clay', 'Grass']
    results = []
    
    print("Simulating matchups across surfaces...")
    for surf in surfaces:
        for ht in heights:
            # We simulate a "Standard Elite Matchup"
            # Player 1 is our variable height player
            # Player 2 is a fixed 185cm opponent
            # All other stats are kept at 'average' (diffs = 0)
            
            # Since the predictor uses the internal state, we can pass custom stats
            # to override the defaults for our simulated player
            p1_custom = {'ht': float(ht), 'age': 25.0, 'rank': 50, 'pts': 1500, 'matches': 100}
            p2_custom = {'ht': 185.0, 'age': 25.0, 'rank': 50, 'pts': 1500, 'matches': 100}
            
            # Predict
            # Note: We use 'G' level for highest signal
            prob = predictor.predict("Simulated Player", "Simulated Opponent", 
                                     surface=surf, level="G", 
                                     p1_custom=p1_custom, p2_custom=p2_custom)
            
            results.append({
                'Height': ht,
                'Surface': surf,
                'Win Probability': prob
            })
            
    sim_df = pd.DataFrame(results)
    
    # 3. Plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    sns.lineplot(data=sim_df, x='Height', y='Win Probability', hue='Surface', 
                 linewidth=3, marker='o', palette={'Hard': 'blue', 'Clay': 'orange', 'Grass': 'green'})
    
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Even Matchup')
    plt.title("Model Interrogation: How Height Influences Win Probability", fontsize=18)
    plt.xlabel("Player 1 Height (cm)", fontsize=14)
    plt.ylabel("Win Probability vs. 185cm Opponent", fontsize=14)
    plt.ylim(0.3, 0.7) # Focus on the range of influence
    
    # Annotations
    plt.annotate('The "Sweet Spot"', xy=(190, sim_df[sim_df['Height']==190]['Win Probability'].max()), 
                 xytext=(180, 0.65), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/height_win_simulation.png')
    print("Simulation plot saved to plots/height_win_simulation.png")

if __name__ == "__main__":
    simulate_height_influence()
