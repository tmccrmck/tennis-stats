from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants
YEARS: List[str] = ['2019', '2020', '2021', '2022', '2023', '2024']
DATA_DIR: str = 'data/'
PLOTS_DIR: str = 'plots/'

DEFAULT_TARGETS = [
    "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev",
    "Alexander Zverev", "Taylor Fritz", "Alex De Minaur", "Casper Ruud",
    "Nick Kyrgios", "John Isner", "Reilly Opelka", "Diego Schwartzman",
    "Rafael Nadal", "Roger Federer", "Holger Rune", "Andrey Rublev",
    "Stefanos Tsitsipas", "Hubert Hurkacz", "Grigor Dimitrov"
]

def load_data(years: List[str] = YEARS) -> pd.DataFrame:
    """Loads and concatenates match data for specified years."""
    dfs: List[pd.DataFrame] = []
    for y in years:
        path: str = os.path.join(DATA_DIR, f'atp_matches_{y}.csv')
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs).reset_index(drop=True)

def load_mcp_data(filename: str) -> pd.DataFrame:
    """Loads Match Charting Project data and joins with match metadata."""
    stats_df = pd.read_csv(os.path.join(DATA_DIR, filename), low_memory=False)
    matches_df = pd.read_csv(os.path.join(DATA_DIR, 'charting-m-matches.csv'))
    return stats_df.merge(matches_df[['match_id', 'Player 1', 'Player 2', 'Date', 'Surface']], on='match_id')

def get_player_heights() -> pd.DataFrame:
    """Returns a mapping of player names to their heights."""
    df = load_data()
    w = df[['winner_name', 'winner_ht']].rename(columns={'winner_name': 'name', 'winner_ht': 'ht'})
    l = df[['loser_name', 'loser_ht']].rename(columns={'loser_name': 'name', 'loser_ht': 'ht'})
    return pd.concat([w, l]).drop_duplicates('name').dropna()

def plot_pca_map(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, filename: str, 
                 hue_col: Optional[str] = None, targets: List[str] = DEFAULT_TARGETS):
    """Standardized PCA visualization with annotations."""
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="white")
    
    sns.scatterplot(
        data=df, x='PC1', y='PC2', hue=hue_col, 
        palette='coolwarm' if hue_col == 'ht' else 'Set1',
        s=100, alpha=0.7, edgecolors='gray'
    )
    
    for name in targets:
        if name in df.index:
            plt.annotate(name, (df.loc[name, 'PC1'], df.loc[name, 'PC2']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Plot saved to {os.path.join(PLOTS_DIR, filename)}")

def get_consolidated_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidates winner and loser data."""
    w_cols = {
        'name': 'winner_name', 'ht': 'winner_ht', 'age': 'winner_age',
        'ace': 'w_ace', 'df': 'w_df', 'svpt': 'w_svpt', '1stIn': 'w_1stIn',
        '1stWon': 'w_1stWon', '2ndWon': 'w_2ndWon', 'SvGms': 'w_SvGms',
        'bpSaved': 'w_bpSaved', 'bpFaced': 'w_bpFaced',
        'opp_svpt': 'l_svpt', 'opp_1stWon': 'l_1stWon', 'opp_2ndWon': 'l_2ndWon',
        'opp_SvGms': 'l_SvGms', 'opp_bpSaved': 'l_bpSaved', 'opp_bpFaced': 'l_bpFaced'
    }
    l_cols = {
        'name': 'loser_name', 'ht': 'loser_ht', 'age': 'loser_age',
        'ace': 'l_ace', 'df': 'l_df', 'svpt': 'l_svpt', '1stIn': 'l_1stIn',
        '1stWon': 'l_1stWon', '2ndWon': 'l_2ndWon', 'SvGms': 'l_SvGms',
        'bpSaved': 'l_bpSaved', 'bpFaced': 'l_bpFaced',
        'opp_svpt': 'w_svpt', 'opp_1stWon': 'w_1stWon', 'opp_2ndWon': 'w_2ndWon',
        'opp_SvGms': 'w_SvGms', 'opp_bpSaved': 'w_bpSaved', 'opp_bpFaced': 'w_bpFaced'
    }
    winners = df[list(w_cols.values())].rename(columns={v: k for k, v in w_cols.items()})
    winners['win'] = 1
    losers = df[list(l_cols.values())].rename(columns={v: k for k, v in l_cols.items()})
    losers['win'] = 0
    combined = pd.concat([winners, losers]).dropna(subset=['ht', 'svpt', 'opp_svpt'])
    return combined[combined['svpt'] > 0]
