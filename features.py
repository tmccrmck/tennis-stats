import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Deque
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import joblib
import config

class ScoreParser:
    @staticmethod
    def parse(score_str: Any) -> Tuple[int, int, int, int]:
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
    def __init__(self) -> None:
        self.recent_matches: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=10))
        self.recent_sets: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=20))
        
        # Style Tracking
        self.recent_hold_pct: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        self.recent_break_pct: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        self.recent_bp_clutch: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        
        self.elo_ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
        self.surface_elo_ratings: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1500.0))
        self.player_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'ht': 185.0, 'age': 25.0, 'rank': 100, 'pts': 500, 'hand': 'R', 'matches': 0})
        self.fatigue_log: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def get_fatigue(self, player: str, current_date: datetime) -> float:
        threshold = current_date - timedelta(days=14)
        self.fatigue_log[player] = [entry for entry in self.fatigue_log[player] if entry[0] >= threshold]
        return sum(entry[1] for entry in self.fatigue_log[player])

    def update_history(self, row: pd.Series) -> None:
        w, l, d_str, surf = row['winner_name'], row['loser_name'], row['tourney_date'], row['surface']
        ws, ls, wg, lg = ScoreParser.parse(row['score'])
        curr_date = datetime.strptime(str(d_str), '%Y%m%d')
        
        # 1. Update Elo (Weighted by Score Margin)
        margin = (wg - lg) / max(1, wg + lg)
        mult = 1.0 + margin
        
        expected = 1 / (1 + 10 ** ((self.elo_ratings[l] - self.elo_ratings[w]) / 400))
        change = 32 * mult * (1 - expected)
        self.elo_ratings[w] += change
        self.elo_ratings[l] -= change
        
        s_expected = 1 / (1 + 10 ** ((self.surface_elo_ratings[l][surf] - self.surface_elo_ratings[w][surf]) / 400))
        s_change = 32 * mult * (1 - s_expected)
        self.surface_elo_ratings[w][surf] += s_change
        self.surface_elo_ratings[l][surf] -= s_change

        # 2. Update Stats & Fatigue
        if not pd.isna(row['minutes']):
            self.fatigue_log[w].append((curr_date, float(row['minutes'])))
            self.fatigue_log[l].append((curr_date, float(row['minutes'])))
        
        # Style Metrics
        try:
            # Hold % = (SvGms - (bpFaced - bpSaved)) / SvGms
            w_h = (row['w_SvGms'] - (row['w_bpFaced'] - row['w_bpSaved'])) / max(1, row['w_SvGms'])
            l_h = (row['l_SvGms'] - (row['l_bpFaced'] - row['l_bpSaved'])) / max(1, row['l_SvGms'])
            
            # Break % = (Opponent's bpFaced - Opponent's bpSaved) / Opponent's SvGms
            w_b = (row['l_bpFaced'] - row['l_bpSaved']) / max(1, row['l_SvGms'])
            l_b = (row['w_bpFaced'] - row['w_bpSaved']) / max(1, row['w_SvGms'])
            
            # BP Clutch = bpSaved% + bpConverted%
            w_c = (row['w_bpSaved'] / max(1, row['w_bpFaced'])) + (w_b) # Proxy
            l_c = (row['l_bpSaved'] / max(1, row['l_bpFaced'])) + (l_b)
            
            self.recent_hold_pct[w].append(float(w_h))
            self.recent_hold_pct[l].append(float(l_h))
            self.recent_break_pct[w].append(float(w_b))
            self.recent_break_pct[l].append(float(l_b))
            self.recent_bp_clutch[w].append(float(w_c))
            self.recent_bp_clutch[l].append(float(l_c))
        except: pass
        
        # Basic Player Info & Experience
        self.player_stats[w] = {
            'ht': row['winner_ht'], 'age': row['winner_age'], 
            'rank': row['winner_rank'], 'pts': row['winner_rank_points'],
            'hand': row['winner_hand'], 'matches': self.player_stats[w].get('matches', 0) + 1
        }
        self.player_stats[l] = {
            'ht': row['loser_ht'], 'age': row['loser_age'], 
            'rank': row['loser_rank'], 'pts': row['loser_rank_points'],
            'hand': row['loser_hand'], 'matches': self.player_stats[l].get('matches', 0) + 1
        }
        
        for _ in range(ws): self.recent_sets[w].append(1); self.recent_sets[l].append(0)
        for _ in range(ls): self.recent_sets[w].append(0); self.recent_sets[l].append(1)
        self.recent_matches[w].append(1); self.recent_matches[l].append(0)

    def extract_features(self, p1: str, p2: str, date_str: Any, surface: str) -> Dict[str, float]:
        curr_date = datetime.strptime(str(date_str), '%Y%m%d')
        
        def g_avg(d: Dict, p: str, default: float) -> float: return np.mean(d[p]) if d[p] else default
        
        return {
            'elo_diff': self.elo_ratings[p1] - self.elo_ratings[p2],
            'surf_elo_diff': self.surface_elo_ratings[p1][surface] - self.surface_elo_ratings[p2][surface],
            'fatigue_diff': self.get_fatigue(p1, curr_date) - self.get_fatigue(p2, curr_date),
            'set_wr_diff': g_avg(self.recent_sets, p1, 0.5) - g_avg(self.recent_sets, p2, 0.5),
            'hold_pct_diff': g_avg(self.recent_hold_pct, p1, 0.8) - g_avg(self.recent_hold_pct, p2, 0.8),
            'break_pct_diff': g_avg(self.recent_break_pct, p1, 0.2) - g_avg(self.recent_break_pct, p2, 0.2),
            'bp_clutch_diff': g_avg(self.recent_bp_clutch, p1, 0.8) - g_avg(self.recent_bp_clutch, p2, 0.8),
            'exp_diff': float(self.player_stats[p1].get('matches', 0) - self.player_stats[p2].get('matches', 0))
        }

    def save(self, path: str = config.STATE_PATH) -> None:
        state = {
            'elo': dict(self.elo_ratings),
            'surf_elo': {k: dict(v) for k, v in self.surface_elo_ratings.items()},
            'player_stats': dict(self.player_stats),
            'recent_hold_pct': {k: list(v) for k, v in self.recent_hold_pct.items()},
            'recent_break_pct': {k: list(v) for k, v in self.recent_break_pct.items()},
            'recent_bp_clutch': {k: list(v) for k, v in self.recent_bp_clutch.items()},
            'fatigue_log': {k: list(v) for k, v in self.fatigue_log.items()},
            'recent_sets': {k: list(v) for k, v in self.recent_sets.items()}
        }
        joblib.dump(state, path)
