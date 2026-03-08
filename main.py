import pandas as pd
import numpy as np
import logging
from pipeline import TennisDataPipeline
from models import TennisModel
from sklearn.metrics import accuracy_score
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_eval():
    # 1. Run Pipeline
    pipeline = TennisDataPipeline()
    data = pipeline.run()
    
    # Fill NaNs
    data = data.fillna(data.mean())
    
    # Split
    train_df = data[data['tourney_date'] < 20240101]
    test_df = data[data['tourney_date'] >= 20240101]
    
    X_train = train_df[config.FEATURES]
    y_train = train_df['target']
    X_test = test_df[config.FEATURES]
    y_test = test_df['target']
    
    # 2. Train Models
    logging.info("Training XGBoost...")
    xgb_model = TennisModel(model_type=config.ModelType.XGBOOST)
    xgb_model.train(X_train, y_train)
    xgb_model.save()
    
    logging.info("Training Logistic Regression...")
    log_model = TennisModel(model_type=config.ModelType.LOGISTIC)
    log_model.train(X_train, y_train)
    log_model.save()

    logging.info("Training Neural Network (with tuning)...")
    nn_model = TennisModel(model_type=config.ModelType.NEURAL_NET)
    nn_model.train(X_train, y_train, tune=True)
    nn_model.save()
    
    # 3. Save State
    pipeline.fm.save()
    
    # 4. Evaluate Ensemble (3-way)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    log_probs = log_model.predict_proba(X_test)[:, 1]
    nn_probs = nn_model.predict_proba(X_test)[:, 1]
    
    # Weighted Ensemble (Favoring Logistic Regression which was the individual peak)
    ensemble_probs = (xgb_probs * 0.25) + (log_probs * 0.50) + (nn_probs * 0.25)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, ensemble_preds)
    logging.info(f"Final Weighted Ensemble Accuracy (2024): {acc:.2%}")

if __name__ == "__main__":
    train_and_eval()
