import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, get_consolidated_player_df

def analyze_handedness_serve():
    print("Analyzing service efficiency of Lefties vs. Righties...")
    df = load_data()
    
    # Filter for known handedness and stats
    pm_df = get_consolidated_player_df(df)
    pm_df = pm_df[pm_df['hand'].isin(['R', 'L'])]
    
    # Calculate Service Metrics
    pm_df['ace_pct'] = (pm_df['ace'] / pm_df['svpt']) * 100
    pm_df['first_won_pct'] = (pm_df['1stWon'] / pm_df['1stIn']) * 100
    pm_df['hold_pct'] = (pm_df['SvGms'] - (pm_df['bpFaced'] - pm_df['bpSaved'])) / pm_df['SvGms'] * 100
    
    # Group by hand
    serve_stats = pm_df.groupby('hand').agg({
        'ace_pct': 'mean',
        'first_won_pct': 'mean',
        'hold_pct': 'mean'
    }).T
    
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    serve_stats.plot(kind='bar', figsize=(12, 7), color=['#ff7f0e', '#1f77b4'])
    
    plt.title("Service DNA: Do Lefties Have a Statistical Advantage on Serve? (2019-2024)", fontsize=18)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    
    # Add data labels
    for i, p in enumerate(plt.gca().patches):
        plt.gca().annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    plt.savefig('plots/handedness_serve_comparison.png')
    print("Handedness serve comparison saved to plots/handedness_serve_comparison.png")
    
    # Surface-specific hold rate (The ultimate lefty weapon test)
    surface_stats = pm_df.groupby(['hand', 'surface'])['hold_pct'].mean().unstack().T
    
    plt.figure(figsize=(12, 7))
    surface_stats.plot(kind='bar', color=['#ff7f0e', '#1f77b4'])
    plt.title("Hold % by Surface: Lefties vs Righties", fontsize=16)
    plt.ylabel("Hold %")
    plt.ylim(70, 85)
    plt.tight_layout()
    plt.savefig('plots/handedness_hold_by_surface.png')
    print("Surface hold comparison saved to plots/handedness_hold_by_surface.png")

if __name__ == "__main__":
    analyze_handedness_serve()
