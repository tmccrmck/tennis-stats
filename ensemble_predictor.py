import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import accuracy_score
from train_model import TennisDataPipeline, ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsemblePredictor:
    def __init__(self, xgb_path='tennis_model.joblib', log_path='logistic_model.joblib', scaler_path='scaler.joblib'):
        logging.info("Loading models for Ensemble...")
        self.xgb_model = joblib.load(xgb_path)
        self.log_model = joblib.load(log_path)
        self.scaler = joblib.load(scaler_path)
        
        self.state = joblib.load('feature_state.joblib')
        self.le_surface = joblib.load('le_surface.joblib')
        self.le_level = joblib.load('le_level.joblib')
        
        # X_train columns order (required for consistency)
        self.feature_cols = ['surface', 'tourney_level', 'p1_rank', 'p1_pts', 'p1_ht', 'p1_age', 
                             'p2_rank', 'p2_pts', 'p2_ht', 'p2_age', 'elo_diff', 'surf_elo_diff', 
                             'h2h_diff', 'surf_h2h_diff', 'fatigue_diff', 'set_wr_diff', 
                             'dom_ratio_diff', 'rank_diff', 'pts_diff']

    def predict_match(self, p1, p2, surface, level, weight_log=0.6, weight_xgb=0.4):
        def get_stat(p): return self.state['player_stats'].get(p, {'ht': 185, 'age': 25, 'rank': 100, 'pts': 500})
        s1, s2 = get_stat(p1), get_stat(p2)
        
        e_diff = self.state['elo'].get(p1, 1500) - self.state['elo'].get(p2, 1500)
        se_diff = self.state['surf_elo'].get(p1, {}).get(surface, 1500) - self.state['surf_elo'].get(p2, {}).get(surface, 1500)
        
        p_str = str(tuple(sorted([p1, p2])))
        sh2h = self.state['surf_h2h'].get(p_str, {}).get(p1, 0) - self.state['surf_h2h'].get(p_str, {}).get(p2, 0)
        
        def g_dom(p): return np.mean(self.state['recent_dominance'].get(p, [])) if self.state['recent_dominance'].get(p) else 1.0
        
        feats = pd.DataFrame([{
            'surface': self.le_surface.transform([surface])[0],
            'tourney_level': self.le_level.transform([level])[0],
            'p1_rank': s1['rank'], 'p1_pts': s1['pts'], 'p1_ht': s1['ht'], 'p1_age': s1['age'],
            'p2_rank': s2['rank'], 'p2_pts': s2['pts'], 'p2_ht': s2['ht'], 'p2_age': s2['age'],
            'elo_diff': e_diff, 'surf_elo_diff': se_diff,
            'h2h_diff': 0, 'surf_h2h_diff': sh2h, 'fatigue_diff': 0, 'set_wr_diff': 0, 
            'dom_ratio_diff': g_dom(p1) - g_dom(p2),
            'rank_diff': s2['rank'] - s1['rank'], 'pts_diff': s1['pts'] - s2['pts']
        }])[self.feature_cols].fillna(0)
        
        # XGBoost Prediction
        # For CalibratedClassifierCV, we call predict_proba
        xgb_prob = self.xgb_model.predict_proba(feats)[0][1]
        
        # Logistic Prediction
        feats_scaled = self.scaler.transform(feats)
        log_prob = self.log_model.predict_proba(feats_scaled)[0][1]
        
        # Weighted Average
        final_prob = (xgb_prob * weight_xgb) + (log_prob * weight_log)
        
        print(f"\nEnsemble Match Prediction: {p1} vs {p2} on {surface}")
        print(f"Confidence {p1} wins: {final_prob:.1%}")
        return final_prob

    def evaluate_ensemble(self, test_df, weight_log=0.6, weight_xgb=0.4):
        logging.info("Evaluating Ensemble on test data...")
        X_test = test_df.drop(['target', 'tourney_date'], axis=1)
        y_test = test_df['target']
        
        # XGB Probabilities
        xgb_probs = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Logistic Probabilities
        X_test_scaled = self.scaler.transform(X_test)
        log_probs = self.log_model.predict_proba(X_test_scaled)[:, 1]
        
        # Ensemble Probabilities
        ensemble_probs = (xgb_probs * weight_xgb) + (log_probs * weight_log)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, ensemble_preds)
        logging.info(f"Ensemble Accuracy (2024): {acc:.2%}")
        return acc

if __name__ == "__main__":
    # 1. Load data for evaluation
    pipeline = TennisDataPipeline()
    symmetrized_data = pipeline.load_and_process()
    test_data = symmetrized_data[symmetrized_data['tourney_date'] >= 20240101].fillna(symmetrized_data.mean())
    
    # 2. Run Ensemble
    ensemble = EnsemblePredictor()
    ensemble.evaluate_ensemble(test_data)
    
    # 3. Example Prediction
    ensemble.predict_match("Jannik Sinner", "Carlos Alcaraz", "Hard", "G")
    ensemble.predict_match("Novak Djokovic", "Jannik Sinner", "Hard", "G")
