from typing import List
from enum import Enum, auto

class ModelType(Enum):
    XGBOOST = auto()
    LOGISTIC = auto()
    NEURAL_NET = auto()

# Data Configuration
YEARS: List[str] = ['2019', '2020', '2021', '2022', '2023', '2024']
DATA_DIR: str = 'data/'
PLOTS_DIR: str = 'plots/'

# Feature Definitions
# These are the columns the models will actually see
FEATURES: List[str] = [
    'surface', 
    'tourney_level', 
    'p1_ht', 
    'p1_age', 
    'p2_ht', 
    'p2_age',
    'p1_hand', # New
    'p2_hand', # New
    'elo_diff', 
    'surf_elo_diff', 
    'fatigue_diff', 
    'set_wr_diff', 
    'hold_pct_diff', # Replaces part of dom_ratio
    'break_pct_diff', # Replaces part of dom_ratio
    'bp_clutch_diff', # New (Saved + Converted)
    'exp_diff', # New (Matches played)
    'rank_diff', 
    'pts_diff'
]

# Model Persistence
XGB_MODEL_PATH: str = 'tennis_model.joblib'
LOG_MODEL_PATH: str = 'logistic_model.joblib'
NN_MODEL_PATH: str = 'nn_model.joblib'
SCALER_PATH: str = 'scaler.joblib'
STATE_PATH: str = 'feature_state.joblib'
LE_SURFACE_PATH: str = 'le_surface.joblib'
LE_LEVEL_PATH: str = 'le_level.joblib'

# Hyperparameters
XGB_PARAMS = {
    'n_estimators': 300, 
    'max_depth': 4, 
    'learning_rate': 0.03, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'gamma': 0.1, 
    'random_state': 42, 
    'eval_metric': 'logloss'
}

NN_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001, # L2 penalty
    'learning_rate_init': 0.001,
    'max_iter': 20,
    'random_state': 42
}
