import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import config
from features import FeatureManager, ScoreParser

class TennisDataPipeline:
    def __init__(self, years: List[str] = config.YEARS) -> None:
        self.years = years
        self.fm = FeatureManager()
        self.le_surf = LabelEncoder()
        self.le_level = LabelEncoder()

    def run(self) -> pd.DataFrame:
        logging.info(f"Processing years: {self.years}")
        dfs = [pd.read_csv(f'data/atp_matches_{y}.csv') for y in self.years]
        df = pd.concat(dfs).dropna(subset=['winner_name', 'loser_name', 'surface', 'winner_rank', 'loser_rank'])
        df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        
        features_list = []
        for _, row in df.iterrows():
            features_list.append(self.fm.extract_features(row['winner_name'], row['loser_name'], row['tourney_date'], row['surface']))
            self.fm.update_history(row)
            
        df_feats = pd.concat([df, pd.DataFrame(features_list)], axis=1)
        return self.symmetrize(df_feats)

    def symmetrize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Base Diffs
        df['rank_diff'] = df['loser_rank'] - df['winner_rank']
        df['pts_diff'] = df['winner_rank_points'] - df['loser_rank_points']
        
        # 2. Encode Handedness (Simple mapping: R=1, L=-1, U=0)
        hand_map = {'R': 1, 'L': -1, 'U': 0}
        df['w_hand_val'] = df['winner_hand'].map(hand_map).fillna(0)
        df['l_hand_val'] = df['loser_hand'].map(hand_map).fillna(0)

        # 3. P1 Wins
        df_p1 = df.copy(); df_p1['target'] = 1
        p1_cols = {
            'winner_ht': 'p1_ht', 'winner_age': 'p1_age', 'w_hand_val': 'p1_hand',
            'loser_ht': 'p2_ht', 'loser_age': 'p2_age', 'l_hand_val': 'p2_hand'
        }
        df_p1 = df_p1.rename(columns=p1_cols)
        
        # 4. P2 Wins
        df_p2 = df.copy(); df_p2['target'] = 0
        p2_cols = {
            'loser_ht': 'p1_ht', 'loser_age': 'p1_age', 'l_hand_val': 'p1_hand',
            'winner_ht': 'p2_ht', 'winner_age': 'p2_age', 'w_hand_val': 'p2_hand'
        }
        df_p2 = df_p2.rename(columns=p2_cols)
        
        # Invert diffs for P2
        diff_cols = ['elo_diff', 'surf_elo_diff', 'fatigue_diff', 'set_wr_diff', 'hold_pct_diff', 'break_pct_diff', 'bp_clutch_diff', 'exp_diff', 'rank_diff', 'pts_diff']
        for col in diff_cols:
            if col in df_p2.columns:
                df_p2[col] = -df_p2[col]
            
        # Select and Combine
        cols = config.FEATURES + ['tourney_date', 'target']
        symm = pd.concat([df_p1[cols].iloc[::2], df_p2[cols].iloc[1::2]]).sort_values('tourney_date')
        
        # Encoding
        symm['surface'] = self.le_surf.fit_transform(symm['surface'])
        symm['tourney_level'] = self.le_level.fit_transform(symm['tourney_level'])
        
        # Save Encoders
        joblib.dump(self.le_surf, config.LE_SURFACE_PATH)
        joblib.dump(self.le_level, config.LE_LEVEL_PATH)
        
        return symm
