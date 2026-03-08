import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional
from features import FeatureManager
from models import TennisModel
import config

class Predictor:
    def __init__(self) -> None:
        self.xgb = TennisModel(model_type=config.ModelType.XGBOOST)
        self.log = TennisModel(model_type=config.ModelType.LOGISTIC)
        self.nn = TennisModel(model_type=config.ModelType.NEURAL_NET)
        self.xgb.load()
        self.log.load()
        self.nn.load()
        
        self.state = joblib.load(config.STATE_PATH)
        self.le_surf = joblib.load(config.LE_SURFACE_PATH)
        self.le_lvl = joblib.load(config.LE_LEVEL_PATH)

    def predict(self, p1: str, p2: str, surface: str, level: str, p1_custom: Optional[Dict] = None, p2_custom: Optional[Dict] = None) -> float:
        def get_stat(p: str, custom: Optional[Dict]):
            base = self.state['player_stats'].get(p, {'ht': 185.0, 'age': 25.0, 'rank': 100, 'pts': 500, 'hand': 'R', 'matches': 0})
            if custom: base.update(custom)
            return base

        s1, s2 = get_stat(p1, p1_custom), get_stat(p2, p2_custom)
        e_diff = self.state['elo'].get(p1, 1500.0) - self.state['elo'].get(p2, 1500.0)
        se_diff = self.state['surf_elo'].get(p1, {}).get(surface, 1500.0) - self.state['surf_elo'].get(p2, {}).get(surface, 1500.0)
        
        # Style mapping
        hand_map = {'R': 1, 'L': -1, 'U': 0}
        p1_h = hand_map.get(s1.get('hand', 'R'), 1)
        p2_h = hand_map.get(s2.get('hand', 'R'), 1)
        
        def g_avg(d_key: str, p: str, default: float) -> float:
            vals = self.state.get(d_key, {}).get(p, [])
            return np.mean(vals) if vals else default
        
        feats = pd.DataFrame([{
            'surface': self.le_surf.transform([surface])[0],
            'tourney_level': self.le_lvl.transform([level])[0],
            'p1_ht': s1['ht'], 'p1_age': s1['age'], 'p2_ht': s2['ht'], 'p2_age': s2['age'],
            'p1_hand': p1_h, 'p2_hand': p2_h,
            'elo_diff': e_diff, 'surf_elo_diff': se_diff, 'fatigue_diff': 0.0, 
            'set_wr_diff': g_avg('recent_sets', p1, 0.5) - g_avg('recent_sets', p2, 0.5),
            'hold_pct_diff': g_avg('recent_hold_pct', p1, 0.8) - g_avg('recent_hold_pct', p2, 0.8),
            'break_pct_diff': g_avg('recent_break_pct', p1, 0.2) - g_avg('recent_break_pct', p2, 0.2),
            'bp_clutch_diff': g_avg('recent_bp_clutch', p1, 0.8) - g_avg('recent_bp_clutch', p2, 0.8),
            'exp_diff': float(s1.get('matches', 0) - s2.get('matches', 0)),
            'rank_diff': float(s2['rank'] - s1['rank']), 'pts_diff': float(s1['pts'] - s2['pts'])
        }])[config.FEATURES].fillna(0.0)
        
        xgb_prob = self.xgb.predict_proba(feats)[0][1]
        log_prob = self.log.predict_proba(feats)[0][1]
        nn_prob = self.nn.predict_proba(feats)[0][1]
        
        # Weighted Ensemble (50% Logistic, 25% XGB, 25% NN)
        return (xgb_prob * 0.25) + (log_prob * 0.50) + (nn_prob * 0.25)
