import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.calibration import calibration_curve
import sys
import os

# Add root to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import TennisDataPipeline
from features import FeatureManager, ScoreParser
from prediction import Predictor
import config

# Set visual style
sns.set_theme(style="whitegrid")

def plot_elo_timeseries():
    print("Generating Elo Time Series...")
    pipeline = TennisDataPipeline()
    years = config.YEARS
    dfs = [pd.read_csv(f'data/atp_matches_{y}.csv') for y in years]
    df = pd.concat(dfs).dropna(subset=['winner_name', 'loser_name', 'surface', 'winner_rank', 'loser_rank'])
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    
    fm = FeatureManager()
    players_to_track = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic"]
    history = {p: [] for p in players_to_track}
    dates = {p: [] for p in players_to_track}
    
    for _, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']
        fm.update_history(row)
        
        for p in players_to_track:
            if w == p or l == p:
                history[p].append(fm.elo_ratings[p])
                dates[p].append(pd.to_datetime(str(row['tourney_date']), format='%Y%m%d'))

    plt.figure(figsize=(14, 7))
    for p in players_to_track:
        plt.plot(dates[p], history[p], label=p, linewidth=2.5)
    
    plt.title("The Rise of Sinner: Elo Rating Comparison (2019-2024)", fontsize=18)
    plt.ylabel("Elo Rating", fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('plots/elo_timeseries.png')
    print("Saved plots/elo_timeseries.png")

def plot_radar_chart():
    print("Generating Sinner vs. Alcaraz Radar Chart...")
    state = joblib.load(config.STATE_PATH)
    
    players = ["Jannik Sinner", "Carlos Alcaraz"]
    categories = ['Hard Elo', 'Clay Elo', 'Grass Elo', 'Hold %', 'Break %']
    N = len(categories)
    
    def get_player_data(p):
        hard = state['surf_elo'].get(p, {}).get('Hard', 1500)
        clay = state['surf_elo'].get(p, {}).get('Clay', 1500)
        grass = state['surf_elo'].get(p, {}).get('Grass', 1500)
        # Hold and Break are now separate
        hold = np.mean(state.get('recent_hold_pct', {}).get(p, [0.8]))
        break_pct = np.mean(state.get('recent_break_pct', {}).get(p, [0.2]))
        return [hard, clay, grass, hold * 2000, break_pct * 8000] # Scaled for visibility

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for p in players:
        values = get_player_data(p)
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=p)
        ax.fill(angles, values, alpha=0.25)
    
    plt.xticks(angles[:-1], categories, fontsize=12)
    plt.title("Head-to-Head Attributes: Sinner vs Alcaraz", size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig('plots/sinner_alcaraz_radar.png')
    print("Saved plots/sinner_alcaraz_radar.png")

def plot_calibration_curve():
    print("Generating Reliability Diagram (Calibration Curve)...")
    predictor = Predictor()
    pipeline = TennisDataPipeline()
    symmetrized_data = pipeline.run()
    test_data = symmetrized_data[symmetrized_data['tourney_date'] >= 20240101].fillna(symmetrized_data.mean())
    
    X_test = test_data[config.FEATURES]
    y_test = test_data['target']
    
    # Ensemble probabilities using the common predictor logic
    probs = []
    for _, row in X_test.iterrows():
        # Build raw features from the test row
        xgb_p = predictor.xgb.predict_proba(pd.DataFrame([row]))[0][1]
        log_p = predictor.log.predict_proba(pd.DataFrame([row]))[0][1]
        nn_p = predictor.nn.predict_proba(pd.DataFrame([row]))[0][1]
        probs.append((xgb_p * 0.25) + (log_p * 0.50) + (nn_p * 0.25))
    
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Weighted Ensemble")
    
    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives", fontsize=14)
    plt.title("Reliability Diagram: Weighted Predictor Calibration", fontsize=18)
    plt.legend(loc="lower right")
    plt.savefig('plots/calibration_curve.png')
    print("Saved plots/calibration_curve.png")

if __name__ == "__main__":
    plot_elo_timeseries()
    plot_radar_chart()
    plot_calibration_curve()
