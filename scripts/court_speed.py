import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def calculate_court_speed():
    print("Calculating Statistical Speed Index (SSI) for ATP Tournaments...")
    df = load_data()
    
    # 1. Consolidate into player-matches to get Hold rates
    pm_df = get_consolidated_player_df(df)
    
    # Add tournament name back to pm_df
    # We'll merge with the original df to get the mapping
    tourney_map = df[['match_num', 'tourney_id', 'tourney_name', 'surface']].drop_duplicates()
    # Note: This is simplified, in a real scenario we'd use a unique match ID
    
    # Calculate Hold Rate per Tournament
    # holds = SvGms - (bpFaced - bpSaved)
    pm_df['holds'] = pm_df['SvGms'] - (pm_df['bpFaced'] - pm_df['bpSaved'])
    
    # Group by tournament
    # We'll use the original match-level df for aces and BP frequency to avoid double counting
    match_df = df.dropna(subset=['w_ace', 'l_ace', 'w_bpFaced', 'l_bpFaced', 'w_SvGms', 'l_SvGms'])
    
    t_stats = match_df.groupby('tourney_name').agg({
        'surface': 'first',
        'w_ace': 'sum',
        'l_ace': 'sum',
        'w_bpFaced': 'sum',
        'l_bpFaced': 'sum',
        'w_SvGms': 'sum',
        'l_SvGms': 'sum',
        'match_num': 'count'
    })
    
    # Filter for significant data
    t_stats = t_stats[t_stats['match_num'] >= 100]
    
    # Metrics
    t_stats['ace_per_match'] = (t_stats['w_ace'] + t_stats['l_ace']) / t_stats['match_num']
    t_stats['bp_per_match'] = (t_stats['w_bpFaced'] + t_stats['l_bpFaced']) / t_stats['match_num']
    
    # Hold Rate
    # Note: winner_holds + loser_holds / total_svgms
    # We need to approximate this from the aggregate
    # This is a bit complex with the current agg, let's simplify:
    # We'll use the ratio of (Total SvGms - Total Breaks) / Total SvGms
    total_breaks = (t_stats['w_bpFaced'] - 0) # This is wrong, let's use player-match df properly
    
    # RE-DOING AGGREGATION PROPERLY
    # Join tourney_name onto player-match df
    pm_df = pm_df.merge(df[['winner_name', 'loser_name', 'tourney_name', 'tourney_date']].drop_duplicates(), 
                        left_on=['name'], right_on=['winner_name'], how='inner') # This is getting complex
    
    # SIMPLIFIED APPROACH: Use match-level data for all speed components
    t_stats = match_df.groupby('tourney_name').agg({
        'surface': 'first',
        'w_ace': 'mean', 'l_ace': 'mean',
        'w_bpFaced': 'mean', 'l_bpFaced': 'mean',
        'w_bpSaved': 'mean', 'l_bpSaved': 'mean',
        'w_SvGms': 'mean', 'l_SvGms': 'mean',
        'match_num': 'count'
    })
    t_stats = t_stats[t_stats['match_num'] >= 80]
    
    t_stats['avg_aces'] = t_stats['w_ace'] + t_stats['l_ace']
    t_stats['avg_bp'] = t_stats['w_bpFaced'] + t_stats['l_bpFaced']
    
    # Hold % approximation
    w_hold_pct = (t_stats['w_SvGms'] - (t_stats['w_bpFaced'] - t_stats['w_bpSaved'])) / t_stats['w_SvGms']
    l_hold_pct = (t_stats['l_SvGms'] - (t_stats['l_bpFaced'] - t_stats['l_bpSaved'])) / t_stats['l_SvGms']
    t_stats['hold_pct'] = (w_hold_pct + l_hold_pct) / 2
    
    # 2. Calculate SSI
    # Higher aces + higher hold% + lower BP = Faster
    t_stats['raw_ssi'] = (t_stats['avg_aces'] * t_stats['hold_pct']) / np.maximum(1, t_stats['avg_bp'])
    
    # Normalize to 0-100
    min_val = t_stats['raw_ssi'].min()
    max_val = t_stats['raw_ssi'].max()
    t_stats['ssi'] = 100 * (t_stats['raw_ssi'] - min_val) / (max_val - min_val)
    
    t_stats = t_stats.sort_values('ssi', ascending=False)
    
    # 3. Plot
    plt.figure(figsize=(12, 15))
    sns.set_theme(style="whitegrid")
    
    # Top 40 tournaments
    plot_df = t_stats.head(40)
    
    colors = {'Hard': 'blue', 'Clay': 'orange', 'Grass': 'green', 'Carpet': 'red'}
    sns.barplot(data=plot_df, x='ssi', y=plot_df.index, hue='surface', palette=colors)
    
    plt.title("ATP Statistical Speed Index (SSI): Ranking Tournament Pace (2019-2024)", fontsize=18)
    plt.xlabel("Speed Index (0 = Slowest, 100 = Fastest)", fontsize=12)
    plt.ylabel("Tournament", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/tournament_speed_ranking.png')
    print("Speed ranking saved to plots/tournament_speed_ranking.png")

if __name__ == "__main__":
    calculate_court_speed()
