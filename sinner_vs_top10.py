import joblib
import pandas as pd
import numpy as np
from train_model import MatchPredictor

def analyze_sinner_dominance():
    # Load predictor and state
    predictor = MatchPredictor()
    state = joblib.load('feature_state.joblib')
    
    # Get latest player stats and sort by rank
    players = []
    for name, stats in state['player_stats'].items():
        players.append({'name': name, 'rank': stats['rank']})
    
    top_players = pd.DataFrame(players).sort_values('rank').head(11) # Get top 11 in case Sinner is there
    top_10_opponents = top_players[top_players['name'] != 'Jannik Sinner'].head(10)
    
    print("Jannik Sinner vs. Current Top 10 (Hard Court, Grand Slam Level)")
    print("-" * 60)
    print(f"{'Opponent':<25} | {'Rank':<5} | {'Sinner Win Prob'}")
    print("-" * 60)
    
    total_prob = 0
    for _, row in top_10_opponents.iterrows():
        opponent = row['name']
        rank = row['rank']
        
        # Build features for prediction
        def get_stat(p): return state['player_stats'].get(p, {'ht': 185, 'age': 25, 'rank': 100, 'pts': 500})
        s1, s2 = get_stat("Jannik Sinner"), get_stat(opponent)
        e_diff = state['elo'].get("Jannik Sinner", 1500) - state['elo'].get(opponent, 1500)
        se_diff = state['surf_elo'].get("Jannik Sinner", {}).get("Hard", 1500) - state['surf_elo'].get(opponent, {}).get("Hard", 1500)
        
        pair_str = str(tuple(sorted(["Jannik Sinner", opponent])))
        sh2h = state['surf_h2h'].get(pair_str, {}).get("Jannik Sinner", 0) - state['surf_h2h'].get(pair_str, {}).get(opponent, 0)
        
        def g_dom(p): return np.mean(state['recent_dominance'].get(p, [])) if state['recent_dominance'].get(p) else 1.0
        
        feats = pd.DataFrame([{
            'surface': predictor.le_surface.transform(["Hard"])[0],
            'tourney_level': predictor.le_level.transform(["G"])[0],
            'p1_rank': s1['rank'], 'p1_pts': s1['pts'], 'p1_ht': s1['ht'], 'p1_age': s1['age'],
            'p2_rank': s2['rank'], 'p2_pts': s2['pts'], 'p2_ht': s2['ht'], 'p2_age': s2['age'],
            'elo_diff': e_diff, 'surf_elo_diff': se_diff,
            'h2h_diff': 0, 'surf_h2h_diff': sh2h, 'fatigue_diff': 0, 'set_wr_diff': 0, 
            'dom_ratio_diff': g_dom("Jannik Sinner") - g_dom(opponent),
            'rank_diff': s2['rank'] - s1['rank'], 'pts_diff': s1['pts'] - s2['pts']
        }])
        
        expected_order = ['surface', 'tourney_level', 'p1_rank', 'p1_pts', 'p1_ht', 'p1_age', 
                          'p2_rank', 'p2_pts', 'p2_ht', 'p2_age', 'elo_diff', 'surf_elo_diff', 
                          'h2h_diff', 'surf_h2h_diff', 'fatigue_diff', 'set_wr_diff', 
                          'dom_ratio_diff', 'rank_diff', 'pts_diff']
        feats = feats[expected_order]
        
        prob = predictor.model.predict_proba(feats)[0][1]
        total_prob += prob
        print(f"{opponent:<25} | {int(rank):<5} | {prob:.1%}")
    
    print("-" * 60)
    print(f"Average Win Probability vs Top 10: {total_prob/10:.1%}")

if __name__ == "__main__":
    analyze_sinner_dominance()
