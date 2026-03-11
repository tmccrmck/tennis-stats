import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import Predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction import Predictor

def simulate_lefty_giant():
    print("Initializing predictor for 'Lefty Giant' simulation...")
    predictor = Predictor()
    
    # 208cm = 6'10"
    height_cm = 208.0
    surfaces = ['Hard', 'Clay', 'Grass']
    hands = ['R', 'L']
    
    results = []
    
    print("Simulating a 208cm (6'10\") player vs. 185cm Average Elite...")
    for surf in surfaces:
        for hand in hands:
            # Simulated Player (variable hand)
            p1_custom = {
                'ht': height_cm, 
                'hand': hand,
                'age': 25.0, 'rank': 50, 'pts': 1500, 'matches': 100
            }
            # Average Elite Opponent (Righty)
            p2_custom = {
                'ht': 185.0, 'hand': 'R',
                'age': 25.0, 'rank': 50, 'pts': 1500, 'matches': 100
            }
            
            prob = predictor.predict("The Giant", "The Opponent", 
                                     surface=surf, level="G", 
                                     p1_custom=p1_custom, p2_custom=p2_custom)
            
            results.append({
                'Surface': surf,
                'Handedness': 'Left-Handed' if hand == 'L' else 'Right-Handed',
                'Win Probability': prob
            })
            
    df = pd.DataFrame(results)
    
    # Visualization
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(data=df, x='Surface', y='Win Probability', hue='Handedness', palette=['#1f77b4', '#ff7f0e'])
    
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.title("The 'Lefty Giant' Theory: Win Probability of a 208cm (6'10\") Player", fontsize=18)
    plt.ylabel("Win Probability vs. 185cm Elite Opponent")
    plt.ylim(0.4, 0.6)
    
    # Add labels
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig('plots/lefty_giant_simulation.png')
    print("Simulation saved to plots/lefty_giant_simulation.png")
    
    # Text Verdict
    lefty_win = df[df['Handedness'] == 'Left-Handed']['Win Probability'].mean()
    righty_win = df[df['Handedness'] == 'Right-Handed']['Win Probability'].mean()
    diff = lefty_win - righty_win
    
    print(f"\nModel Verdict for a 6'10\" Giant:")
    print(f"  - Average Win Probability as a Righty: {righty_win:.2%}")
    print(f"  - Average Win Probability as a Lefty:  {lefty_win:.2%}")
    print(f"  - The 'Southpaw Boost': {diff:+.2%}")

if __name__ == "__main__":
    simulate_lefty_giant()
