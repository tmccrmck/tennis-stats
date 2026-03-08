import pandas as pd
import numpy as np
import os

# Constants
YEARS = ['2020', '2021', '2022', '2023', '2024']
DATA_DIR = 'data/'
PLOTS_DIR = 'plots/'

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import os

# Constants
YEARS: List[str] = ['2020', '2021', '2022', '2023', '2024']
DATA_DIR: str = 'data/'
PLOTS_DIR: str = 'plots/'

def load_data(years: List[str] = YEARS) -> pd.DataFrame:
    """Loads and concatenates match data for specified years."""
    dfs: List[pd.DataFrame] = []
    for y in years:
        path: str = os.path.join(DATA_DIR, f'atp_matches_{y}.csv')
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs).reset_index(drop=True)

def get_consolidated_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates winner and loser data into a single DataFrame where each row 
    is a player's performance in a match.
    """
    # Columns to extract for the "primary" player and the "opponent"
    w_cols: Dict[str, str] = {
        'name': 'winner_name', 'ht': 'winner_ht', 'age': 'winner_age',
        'ace': 'w_ace', 'df': 'w_df', 'svpt': 'w_svpt', '1stIn': 'w_1stIn',
        '1stWon': 'w_1stWon', '2ndWon': 'w_2ndWon', 'SvGms': 'w_SvGms',
        'bpSaved': 'w_bpSaved', 'bpFaced': 'w_bpFaced',
        'opp_svpt': 'l_svpt', 'opp_1stWon': 'l_1stWon', 'opp_2ndWon': 'l_2ndWon',
        'opp_SvGms': 'l_SvGms', 'opp_bpSaved': 'l_bpSaved', 'opp_bpFaced': 'l_bpFaced'
    }
    
    l_cols: Dict[str, str] = {
        'name': 'loser_name', 'ht': 'loser_ht', 'age': 'loser_age',
        'ace': 'l_ace', 'df': 'l_df', 'svpt': 'l_svpt', '1stIn': 'l_1stIn',
        '1stWon': 'l_1stWon', '2ndWon': 'l_2ndWon', 'SvGms': 'l_SvGms',
        'bpSaved': 'l_bpSaved', 'bpFaced': 'l_bpFaced',
        'opp_svpt': 'w_svpt', 'opp_1stWon': 'w_1stWon', 'opp_2ndWon': 'w_2ndWon',
        'opp_SvGms': 'w_SvGms', 'opp_bpSaved': 'w_bpSaved', 'opp_bpFaced': 'w_bpFaced'
    }
    
    # Extract and rename
    winners: pd.DataFrame = df[list(w_cols.values())].rename(columns={v: k for k, v in w_cols.items()})
    winners['win'] = 1
    
    losers: pd.DataFrame = df[list(l_cols.values())].rename(columns={v: k for k, v in l_cols.items()})
    losers['win'] = 0
    
    # Combine
    combined: pd.DataFrame = pd.concat([winners, losers]).dropna(subset=['ht', 'svpt', 'opp_svpt'])
    return combined[combined['svpt'] > 0]
