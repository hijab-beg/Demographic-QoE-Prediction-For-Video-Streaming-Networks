"""
QoE Prediction Module

This module implements machine learning models for Quality of Experience (QoE)
prediction in adaptive video streaming based on streaming metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class QoEPredictor:
    """
    Machine learning-based QoE prediction for video streaming.
    
    This class implements multiple regression models and provides
    comprehensive evaluation metrics for QoE prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the QoE predictor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'mos',
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            df: Dataset with features and target
            target_col: Name of target column
            drop_cols: Columns to exclude from features
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Default columns to drop
        if drop_cols is None:
            drop_cols = ['streaming_log', 'content', 'encoding_profile', 
                        'device', 'demographic', 'original_mos', 'mos_adjustment']
        
        # Remove columns that don't exist
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Separate features and target
        X = df.drop([target_col] + drop_cols, axis=1, errors='ignore')
        y = df[target_col]
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def initialize_models(self) -> Dict:
        """
        Initialize a set of regression models for comparison.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10, 
                min_samples_split=5,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        self.models = models
        return models
    
    def evaluate_model(
        self, 
        model, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict:
        """
        Train and evaluate a single model.
        
        Args:
            model: Scikit-learn model instance
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'Train R²': r2_score(y_train, y_train_pred),
            'Test R²': r2_score(y_test, y_test_pred),
            'Predictions': y_test_pred
        }
        
        return metrics
    
    def compare_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models and identify the best one.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with comparison results
        """
        if not self.models:
            self.initialize_models()
        
        results = []
        
        print("Evaluating models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
            
            results.append({
                'Model': name,
                'Train RMSE': metrics['Train RMSE'],
                'Test RMSE': metrics['Test RMSE'],
                'Train MAE': metrics['Train MAE'],
                'Test MAE': metrics['Test MAE'],
                'Train R²': metrics['Train R²'],
                'Test R²': metrics['Test R²']
            })
            
            self.results[name] = metrics
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test RMSE')
        
        # Identify best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name}")
        
        return results_df
    
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = 'mos',
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            df: Dataset with features and target
            target_col: Name of target column
            drop_cols: Columns to exclude from features
            test_size: Proportion of test set
            
        Returns:
            Tuple of (results DataFrame, best model metrics)
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, target_col, drop_cols, test_size
        )
        
        # Compare models
        results_df = self.compare_models(X_train, y_train, X_test, y_test)
        
        # Get best model metrics
        best_metrics = self.results[self.best_model_name]
        
        return results_df, best_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_and_evaluate first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_feature_importance(
        self, 
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance from the best model (if applicable).
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        
        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"{self.best_model_name} does not support feature importance.")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.best_model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df.head(top_n)


def main():
    """Example usage of QoE predictor."""
    import os
    
    # Load dataset
    df = pd.read_csv("data/combined_dataset_demographic_augmented.csv")
    
    print("Dataset shape:", df.shape)
    print("\nFeatures:", df.columns.tolist())
    
    # Initialize predictor
    predictor = QoEPredictor(random_state=42)
    
    # Train and evaluate
    results_df, best_metrics = predictor.train_and_evaluate(df)
    
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    
    print(f"\n=== Best Model: {predictor.best_model_name} ===")
    print(f"Test RMSE: {best_metrics['Test RMSE']:.4f}")
    print(f"Test MAE: {best_metrics['Test MAE']:.4f}")
    print(f"Test R²: {best_metrics['Test R²']:.4f}")
    
    # Feature importance (if available)
    X = df.drop(['mos', 'streaming_log', 'content', 'encoding_profile', 
                 'device'], axis=1, errors='ignore')
    feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
    
    importance_df = predictor.get_feature_importance(feature_names)
    if importance_df is not None:
        print("\n=== Top 10 Important Features ===")
        print(importance_df.to_string(index=False))


if __name__ == "__main__":
    main()
