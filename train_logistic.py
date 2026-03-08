import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from train_model import TennisDataPipeline, ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LogisticModelManager(ModelManager):
    def __init__(self, df):
        super().__init__(df)
        self.scaler = StandardScaler()

    def train_and_evaluate(self):
        # 1. Scale Features
        logging.info("Scaling features for Logistic Regression...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # 2. Train Logistic Regression
        logging.info("Training Logistic Regression model...")
        model = LogisticRegression(
            C=1.0, 
            penalty='l2', 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42
        )
        model.fit(X_train_scaled, self.y_train)
        
        # 3. Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(self.y_test, y_pred)
        logging.info(f"Logistic Regression Accuracy (2024): {acc:.2%}")
        
        # 4. Save
        joblib.dump(model, 'logistic_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        
        # 5. Feature Coefficients (Interpretable!)
        coef_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', ascending=False)
        
        print("\nTop 5 Positive Coefficients (Increases P1 Win Probability):")
        print(coef_df.head(5))
        print("\nTop 5 Negative Coefficients (Decreases P1 Win Probability):")
        print(coef_df.tail(5))
        
        return model

if __name__ == "__main__":
    # Use the same pipeline to get the exact same data
    pipeline = TennisDataPipeline()
    symmetrized_data = pipeline.load_and_process()
    
    # Handle NaNs for Logistic Regression
    logging.info("Imputing missing values with column means...")
    symmetrized_data = symmetrized_data.fillna(symmetrized_data.mean())
    
    # Train Logistic Regression
    log_manager = LogisticModelManager(symmetrized_data)
    log_model = log_manager.train_and_evaluate()
