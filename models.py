import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
import config
from typing import Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TorchTennisNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], activation: str = 'relu'):
        super(TorchTennisNN, self).__init__()
        layers = []
        last_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            last_dim = h_dim
            
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TorchModelWrapper:
    """Wraps PyTorch logic to look like a scikit-learn classifier."""
    def __init__(self, input_dim: int, params: dict):
        self.params = params
        self.model = TorchTennisNN(
            input_dim=input_dim, 
            hidden_layers=params.get('hidden_layer_sizes', (64, 32)),
            activation=params.get('activation', 'relu')
        )
        self.epochs = params.get('max_iter', 100)
        self.lr = params.get('learning_rate_init', 0.001)
        self.weight_decay = params.get('alpha', 0.0001) # L2 penalty

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(loader):.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self.model(X_tensor).numpy()
        # Return [P(class 0), P(class 1)] to match sklearn format
        return np.hstack([1 - probs, probs])

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

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
            
            # Use current config params
            params = config.NN_PARAMS.copy()
            
            if tune:
                logging.info("Neural Network tuning skipped for PyTorch migration. Using config defaults.")
            
            self.model = TorchModelWrapper(input_dim=X.shape[1], params=params)
            self.model.fit(X_scaled, y.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        else:
            X = X.values # For XGBoost or non-scaled
            
        if self.model_type == config.ModelType.NEURAL_NET:
            return self.model.predict_proba(X)
        return self.model.predict_proba(X)

    def save(self):
        if self.model_type == config.ModelType.XGBOOST:
            joblib.dump(self.model, config.XGB_MODEL_PATH)
        elif self.model_type == config.ModelType.LOGISTIC:
            joblib.dump(self.model, config.LOG_MODEL_PATH)
            joblib.dump(self.scaler, config.SCALER_PATH)
        elif self.model_type == config.ModelType.NEURAL_NET:
            # We save the model wrapper which contains the torch model and params
            # Note: joblib can save torch objects but it's often better to save state_dicts
            # To keep it consistent with the existing codebase using joblib.load, we dump the wrapper.
            joblib.dump({'model_state': self.model.state_dict(), 'params': self.model.params, 'input_dim': len(config.FEATURES)}, config.NN_MODEL_PATH)
            joblib.dump(self.scaler, config.SCALER_PATH)

    def load(self):
        if self.model_type == config.ModelType.XGBOOST:
            self.model = joblib.load(config.XGB_MODEL_PATH)
        elif self.model_type == config.ModelType.LOGISTIC:
            self.model = joblib.load(config.LOG_MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
        elif self.model_type == config.ModelType.NEURAL_NET:
            data = joblib.load(config.NN_MODEL_PATH)
            self.model = TorchModelWrapper(input_dim=data['input_dim'], params=data['params'])
            self.model.load_state_dict(data['model_state'])
            self.scaler = joblib.load(config.SCALER_PATH)
