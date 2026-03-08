import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict, deque
import re
import logging
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScoreParser:
    @staticmethod
    def parse(score_str):
        if pd.isna(score_str) or any(x in str(score_str) for x in ['RET', 'W/O', 'DEF']):
            return 0, 0, 0, 0
        score_str = re.sub(r'\(\d+\)', '', str(score_str))
        sets = score_str.split(' ')
        w_sets, l_sets, w_games, l_games = 0, 0, 0, 0
        for s in sets:
            if '-' in s:
                try:
                    games = s.split('-')
                    g_w, g_l = int(games[0]), int(games[1])
                    w_games += g_w; l_games += g_l
                    if g_w > g_l: w_sets += 1
                    elif g_l > g_w: l_sets += 1
                except (ValueError, IndexError): continue
        return w_sets, l_sets, w_games, l_games

class FeatureManager:
    def __init__(self):
        self.h2h_store = defaultdict(lambda: defaultdict(int))
        self.surf_h2h_store = defaultdict(lambda: defaultdict(int))
        self.recent_matches = defaultdict(lambda: deque(maxlen=10))
        self.recent_sets = defaultdict(lambda: deque(maxlen=20))
        self.recent_dominance = defaultdict(lambda: deque(maxlen=10))
        self.tourney_fatigue = defaultdict(int)
        self.elo_ratings = defaultdict(lambda: 1500.0)
        self.surface_elo_ratings = defaultdict(lambda: defaultdict(lambda: 1500.0))
        self.k_factor = 32
        self.player_stats = defaultdict(lambda: {'ht': 185.0, 'age': 25.0, 'rank': 100, 'pts': 500})

    def calculate_expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update_elo(self, winner, loser, surface, w_games, l_games):
        margin = (w_games - l_games) / max(1, w_games + l_games)
        multiplier = 1.0 + margin
        w_elo, l_elo = self.elo_ratings[winner], self.elo_ratings[loser]
        expected = self.calculate_expected_score(w_elo, l_elo)
        change = self.k_factor * multiplier * (1 - expected)
        self.elo_ratings[winner] += change
        self.elo_ratings[loser] -= change
        sw_elo, sl_elo = self.surface_elo_ratings[winner][surface], self.surface_elo_ratings[loser][surface]
        s_expected = self.calculate_expected_score(sw_elo, sl_elo)
        s_change = self.k_factor * multiplier * (1 - s_expected)
        self.surface_elo_ratings[winner][surface] += s_change
        self.surface_elo_ratings[loser][surface] -= s_change

    def get_pre_match_features(self, winner, loser, tid, surface):
        pair = tuple(sorted([winner, loser]))
        def get_avg_dom(p): return np.mean(self.recent_dominance[p]) if self.recent_dominance[p] else 1.0
        p1_surf_wins = self.surf_h2h_store[(pair[0], pair[1], surface)][winner]
        p2_surf_wins = self.surf_h2h_store[(pair[0], pair[1], surface)][loser]
        
        return {
            'elo_diff': self.elo_ratings[winner] - self.elo_ratings[loser],
            'surf_elo_diff': self.surface_elo_ratings[winner][surface] - self.surface_elo_ratings[loser][surface],
            'h2h_diff': self.h2h_store[pair][winner] - self.h2h_store[pair][loser],
            'surf_h2h_diff': p1_surf_wins - p2_surf_wins,
            'fatigue_diff': self.tourney_fatigue[(winner, tid)] - self.tourney_fatigue[(loser, tid)],
            'set_wr_diff': (sum(self.recent_sets[winner])/max(1, len(self.recent_sets[winner]))) - 
                           (sum(self.recent_sets[loser])/max(1, len(self.recent_sets[loser]))),
            'dom_ratio_diff': get_avg_dom(winner) - get_avg_dom(loser)
        }

    def update_history(self, row, ws, ls, wg, lg):
        w, l, tid, surf = row['winner_name'], row['loser_name'], row['tourney_id'], row['surface']
        self.update_elo(w, l, surf, wg, lg)
        pair = tuple(sorted([w, l]))
        self.h2h_store[pair][w] += 1
        self.surf_h2h_store[(pair[0], pair[1], surf)][w] += 1
        try:
            w_h = (row['w_SvGms'] - (row['w_bpFaced'] - row['w_bpSaved'])) / max(1, row['w_SvGms'])
            w_b = (row['l_bpFaced'] - row['l_bpSaved']) / max(1, row['l_SvGms'])
            l_h = (row['l_SvGms'] - (row['l_bpFaced'] - row['l_bpSaved'])) / max(1, row['l_SvGms'])
            l_b = (row['w_bpFaced'] - row['w_bpSaved']) / max(1, row['w_SvGms'])
            self.recent_dominance[w].append(w_h + w_b)
            self.recent_dominance[l].append(l_h + l_b)
        except: pass
        self.player_stats[w] = {'ht': row['winner_ht'], 'age': row['winner_age'], 'rank': row['winner_rank'], 'pts': row['winner_rank_points']}
        self.player_stats[l] = {'ht': row['loser_ht'], 'age': row['loser_age'], 'rank': row['loser_rank'], 'pts': row['loser_rank_points']}
        if not pd.isna(row['minutes']):
            self.tourney_fatigue[(w, tid)] += row['minutes']
            self.tourney_fatigue[(l, tid)] += row['minutes']
        for _ in range(ws): self.recent_sets[w].append(1); self.recent_sets[l].append(0)
        for _ in range(ls): self.recent_sets[w].append(0); self.recent_sets[l].append(1)
        self.recent_matches[w].append(1); self.recent_matches[l].append(0)

    def save_state(self, path='feature_state.joblib'):
        state = {
            'h2h': dict(self.h2h_store),
            'surf_h2h': {str(k): dict(v) for k, v in self.surf_h2h_store.items()},
            'elo': dict(self.elo_ratings),
            'surf_elo': {k: dict(v) for k, v in self.surface_elo_ratings.items()},
            'player_stats': dict(self.player_stats),
            'recent_matches': {k: list(v) for k, v in self.recent_matches.items()},
            'recent_sets': {k: list(v) for k, v in self.recent_sets.items()},
            'recent_dominance': {k: list(v) for k, v in self.recent_dominance.items()}
        }
        joblib.dump(state, path)

class TennisDataPipeline:
    def __init__(self, years=['2020', '2021', '2022', '2023', '2024']):
        self.years = years
        self.feature_manager = FeatureManager()
        self.le_surface = LabelEncoder()
        self.le_level = LabelEncoder()

    def load_and_process(self):
        dfs = [pd.read_csv(f'data/atp_matches_{y}.csv') for y in self.years]
        df = pd.concat(dfs).dropna(subset=['winner_name', 'loser_name', 'surface', 'winner_rank', 'loser_rank'])
        df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        features_list = []
        for _, row in df.iterrows():
            features_list.append(self.feature_manager.get_pre_match_features(row['winner_name'], row['loser_name'], row['tourney_id'], row['surface']))
            ws, ls, wg, lg = ScoreParser.parse(row['score'])
            self.feature_manager.update_history(row, ws, ls, wg, lg)
        df_features = pd.concat([df, pd.DataFrame(features_list)], axis=1)
        return self.symmetrize(df_features)

    def symmetrize(self, df):
        base_cols = ['surface', 'tourney_level', 'tourney_date', 'winner_rank', 'winner_rank_points', 
                     'winner_ht', 'winner_age', 'loser_rank', 'loser_rank_points', 'loser_ht', 'loser_age',
                     'elo_diff', 'surf_elo_diff', 'h2h_diff', 'surf_h2h_diff', 'fatigue_diff', 'set_wr_diff', 'dom_ratio_diff']
        data = df[base_cols].copy()
        df_p1 = data.copy(); df_p1['target'] = 1
        df_p1.columns = ['surface', 'tourney_level', 'tourney_date', 'p1_rank', 'p1_pts', 'p1_ht', 'p1_age', 
                         'p2_rank', 'p2_pts', 'p2_ht', 'p2_age', 'elo_diff', 'surf_elo_diff', 'h2h_diff', 'surf_h2h_diff', 
                         'fatigue_diff', 'set_wr_diff', 'dom_ratio_diff', 'target']
        df_p2 = data.copy(); df_p2['target'] = 0
        df_p2.columns = ['surface', 'tourney_level', 'tourney_date', 'p2_rank', 'p2_pts', 'p2_ht', 'p2_age', 
                         'p1_rank', 'p1_pts', 'p1_ht', 'p1_age', 'elo_inv', 'surf_inv', 'h2_inv', 'sh2_inv', 'fat_inv', 
                         'set_inv', 'dom_inv', 'target']
        mapping = {'elo': 'elo_inv', 'surf_elo': 'surf_inv', 'h2h': 'h2_inv', 'surf_h2h': 'sh2_inv', 'fatigue': 'fat_inv', 'set_wr': 'set_inv', 'dom_ratio': 'dom_inv'}
        for target, source in mapping.items():
            df_p2[f'{target}_diff'] = -df_p2[source]
        df_p2 = df_p2.drop(list(mapping.values()), axis=1)
        symm = pd.concat([df_p1.iloc[::2], df_p2.iloc[1::2]]).sort_values(['tourney_date'])
        symm['surface'] = self.le_surface.fit_transform(symm['surface'])
        symm['tourney_level'] = self.le_level.fit_transform(symm['tourney_level'])
        symm['rank_diff'] = symm['p2_rank'] - symm['p1_rank']
        symm['pts_diff'] = symm['p1_pts'] - symm['p2_pts']
        return symm

class ModelManager:
    def __init__(self, df):
        self.train_df = df[df['tourney_date'] < 20240101]
        self.test_df = df[df['tourney_date'] >= 20240101]
        self.X_train = self.train_df.drop(['target', 'tourney_date'], axis=1)
        self.y_train = self.train_df['target']
        self.X_test = self.test_df.drop(['target', 'tourney_date'], axis=1)
        self.y_test = self.test_df['target']

    def train_and_evaluate(self):
        params = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'random_state': 42, 'eval_metric': 'logloss'}
        base_model = xgb.XGBClassifier(**params)
        
        # Wrapping in Calibration
        logging.info("Training Calibrated XGBoost model...")
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        model.fit(self.X_train, self.y_train)
        
        acc = accuracy_score(self.y_test, model.predict(self.X_test))
        logging.info(f"Final Calibrated Model Accuracy (2024): {acc:.2%}")
        joblib.dump(model, 'tennis_model.joblib')
        return model

class MatchPredictor:
    def __init__(self):
        self.model = joblib.load('tennis_model.joblib')
        self.state = joblib.load('feature_state.joblib')
        self.le_surface = joblib.load('le_surface.joblib')
        self.le_level = joblib.load('le_level.joblib')

    def predict(self, p1, p2, surface, level):
        def get_stat(p): return self.state['player_stats'].get(p, {'ht': 185, 'age': 25, 'rank': 100, 'pts': 500})
        s1, s2 = get_stat(p1), get_stat(p2)
        e_diff = self.state['elo'].get(p1, 1500) - self.state['elo'].get(p2, 1500)
        se_diff = self.state['surf_elo'].get(p1, {}).get(surface, 1500) - self.state['surf_elo'].get(p2, {}).get(surface, 1500)
        p_str = str(tuple(sorted([p1, p2])))
        sh2h = self.state['surf_h2h'].get(p_str, {}).get(p1, 0) - self.state['surf_h2h'].get(p_str, {}).get(p2, 0)
        def g_dom(p): return np.mean(self.state['recent_dominance'].get(p, [])) if self.state['recent_dominance'].get(p) else 1.0
        
        feats = pd.DataFrame([{
            'surface': self.le_surface.transform([surface])[0], 'tourney_level': self.le_level.transform([level])[0],
            'p1_rank': s1['rank'], 'p1_pts': s1['pts'], 'p1_ht': s1['ht'], 'p1_age': s1['age'],
            'p2_rank': s2['rank'], 'p2_pts': s2['pts'], 'p2_ht': s2['ht'], 'p2_age': s2['age'],
            'elo_diff': e_diff, 'surf_elo_diff': se_diff, 'h2h_diff': 0, 'surf_h2h_diff': sh2h, 'fatigue_diff': 0, 
            'set_wr_diff': 0, 'dom_ratio_diff': g_dom(p1) - g_dom(p2),
            'rank_diff': s2['rank'] - s1['rank'], 'pts_diff': s1['pts'] - s2['pts']
        }])
        
        # Ensure column order matches X_train exactly
        expected_order = ['surface', 'tourney_level', 'p1_rank', 'p1_pts', 'p1_ht', 'p1_age', 
                          'p2_rank', 'p2_pts', 'p2_ht', 'p2_age', 'elo_diff', 'surf_elo_diff', 
                          'h2h_diff', 'surf_h2h_diff', 'fatigue_diff', 'set_wr_diff', 
                          'dom_ratio_diff', 'rank_diff', 'pts_diff']
        feats = feats[expected_order]
        
        prob = self.model.predict_proba(feats)[0][1]
        print(f"\nCalibrated Match Prediction: {p1} vs {p2} on {surface}")
        print(f"Confidence {p1} wins: {prob:.1%}")
        print(f"Confidence {p2} wins: {1-prob:.1%}")

if __name__ == "__main__":
    if not os.path.exists('tennis_model.joblib'):
        pipeline = TennisDataPipeline()
        data = pipeline.load_and_process()
        ModelManager(data).train_and_evaluate()
        pipeline.feature_manager.save_state()
        joblib.dump(pipeline.le_surface, 'le_surface.joblib')
        joblib.dump(pipeline.le_level, 'le_level.joblib')
    predictor = MatchPredictor()
    predictor.predict("Jannik Sinner", "Carlos Alcaraz", "Hard", "G")
