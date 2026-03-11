import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib
import logging
import config
from typing import Any, Tuple, Optional

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

class TennisModel:
    def __init__(self, model_type: config.ModelType = config.ModelType.XGBOOST):
        self.model_type = model_type
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None

    def train(self, X: pd.DataFrame, y: pd.Series, tune: bool = False):
        if self.model_type == config.ModelType.XGBOOST:
            base = xgb.XGBClassifier(**config.XGB_PARAMS)
            self.model = CalibratedClassifierCV(base, method='sigmoid', cv=5)
            self.model.fit(X, y)
        
        elif self.model_type == config.ModelType.LOGISTIC:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_scaled, y)
            
        elif self.model_type == config.ModelType.NEURAL_NET:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            if tune:
                logging.info("Tuning Neural Network hyperparameters...")
                param_grid = {
                    'hidden_layer_sizes': [(64,), (64, 32), (32, 16), (128, 64, 32)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01],
                    'activation': ['relu', 'tanh']
                }
                tscv = TimeSeriesSplit(n_splits=3)
                search = RandomizedSearchCV(
                    MLPClassifier(max_iter=500, random_state=42),
                    param_distributions=param_grid,
                    n_iter=10, cv=tscv, n_jobs=-1, scoring='accuracy'
                )
                search.fit(X_scaled, y)
                self.model = search.best_estimator_
                logging.info(f"Best NN Params: {search.best_params_}")
            else:
                self.model = MLPClassifier(**config.NN_PARAMS)
                self.model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def save(self):
        if self.model_type == config.ModelType.XGBOOST:
            joblib.dump(self.model, config.XGB_MODEL_PATH)
        elif self.model_type == config.ModelType.LOGISTIC:
            joblib.dump(self.model, config.LOG_MODEL_PATH)
            joblib.dump(self.scaler, config.SCALER_PATH)
        elif self.model_type == config.ModelType.NEURAL_NET:
            joblib.dump(self.model, config.NN_MODEL_PATH)
            joblib.dump(self.scaler, config.SCALER_PATH)

    def load(self):
        if self.model_type == config.ModelType.XGBOOST:
            self.model = joblib.load(config.XGB_MODEL_PATH)
        elif self.model_type == config.ModelType.LOGISTIC:
            self.model = joblib.load(config.LOG_MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
        elif self.model_type == config.ModelType.NEURAL_NET:
            self.model = joblib.load(config.NN_MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
