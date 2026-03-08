import joblib
import pandas as pd
from prediction import Predictor
import config

def analyze_sinner_dominance():
    predictor = Predictor()
    
    # Get latest player stats and sort by rank
    players = []
    for name, stats in predictor.state['player_stats'].items():
        players.append({'name': name, 'rank': stats['rank']})
    
    top_players = pd.DataFrame(players).sort_values('rank').head(11)
    top_10_opponents = top_players[top_players['name'] != 'Jannik Sinner'].head(10)
    
    print("Jannik Sinner vs. Current Top 10 (Hard Court, Grand Slam Level)")
    print("-" * 60)
    print(f"{'Opponent':<25} | {'Rank':<5} | {'Sinner Win Prob'}")
    print("-" * 60)
    
    total_prob = 0
    for _, row in top_10_opponents.iterrows():
        opponent = row['name']
        prob = predictor.predict("Jannik Sinner", opponent, "Hard", "G")
        total_prob += prob
        print(f"{opponent:<25} | {int(row['rank']):<5} | {prob:.1%}")
    
    print("-" * 60)
    print(f"Average Win Probability vs Top 10: {total_prob/10:.1%}")

if __name__ == "__main__":
    analyze_sinner_dominance()
