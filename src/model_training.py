import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib
import time
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

class EarthquakeModelTrainer: 
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
    def train_model(self, model, X_train, y_train, X_test, y_test):
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'training_time': training_time,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        self.initialize_models()
        
        print("\n" + "=" * 60)
        print("Training Multiple ML Models")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\n<> Training {name}...")
            result = self.train_model(model, X_train, y_train, X_test, y_test)
            self.results[name] = result
            
            print(f"  Training time: {result['training_time']:.2f}s")
            print(f"  CV F1-Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
        
        print("\n" + "=" * 60)
        print("All models trained successfully!")
        print("=" * 60)
        
        return self.results
    
    def find_best_model(self):
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        best_score = -1
        best_name = None
        
        for name, result in self.results.items():
            if result['cv_mean'] > best_score:
                best_score = result['cv_mean']
                best_name = name
        
        print("\n" + "=" * 60)
        print("Best Model Selection")
        print("=" * 60)
        print(f"\nBest Model: {best_name}")
        print(f"CV F1-Score: {best_score:.4f}")
        print("=" * 60)
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        return best_name, best_score
    
    def get_param_grid(self, model_name):
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        }
        
        return param_grids.get(model_name, {})
    
    def optimize_best_model(self, X_train, y_train, model_name=None):
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("No best model selected. Run find_best_model() first!")
            model_name = self.best_model_name
        
        print("\n" + "=" * 60)
        print(f"Hyperparameter Optimization - {model_name}")
        print("=" * 60)
        
        param_grid = self.get_param_grid(model_name)
        
        if not param_grid:
            print(f"\nNo parameter grid defined for {model_name}")
            print("Using default model without optimization")
            return self.best_model
        
        print("\n<> Running GridSearchCV...")
        print(f"  Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        base_model = self.models[model_name].__class__
        
        if model_name == 'XGBoost':
            model = base_model(random_state=42, eval_metric='logloss')
        elif model_name == 'SVM':
            model = base_model(random_state=42, probability=True)
        else:
            model = base_model(random_state=42)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        print(f"  Optimization complete in {elapsed_time:.1f}s")
        print(f"\n  Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    - {param}: {value}")
        print(f"\n  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"Optimized {model_name}"
        
        print("\n" + "=" * 60)
        
        return self.best_model
    
    def save_model(self, filepath):
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        joblib.dump(self.best_model, filepath)
        print(f"\nModel saved to: {filepath}")
        
    def compare_models(self):
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'CV F1-Score': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}",
                'Training Time (s)': f"{result['training_time']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['_sort'] = comparison_df['CV F1-Score'].astype(float)
        comparison_df = comparison_df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
        
        return comparison_df

def apply_smote(X_train, y_train):
    print("\n<> Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    
    result = smote.fit_resample(np.array(X_train), np.array(y_train))
    X_resampled = result[0]
    y_resampled = result[1]
    
    print(f"  Original training samples: {len(X_train)}")
    print(f"  Resampled training samples: {len(X_resampled)}")
    
    return X_resampled, y_resampled

def main():
    print("=" * 60)
    print("Earthquake Risk Prediction - Model Training")
    print("=" * 60)
    
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_dir = project_root / "outputs" / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n[1/6] Loading preprocessed data...")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["risk"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["risk"]
    
    feature_names = ['latitude', 'longitude', 'depth', 'geo_distance']
    X_train.columns = feature_names
    X_test.columns = feature_names
    
    print(f"  Training samples: {len(X_train)} with {X_train.shape[1]} features")
    print(f"  Testing samples: {len(X_test)}")
    print("\n  Class distribution in training set:")
    for class_label, count in y_train.value_counts().sort_index().items():
        print(f"    - Class {class_label}: {count} samples")
    
    print("\n[2/6] Handling class imbalance...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    print("\n  Balanced class distribution:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for class_label, count in zip(unique, counts):
        print(f"    - Class {class_label}: {count} samples")
    
    print("\n[3/6] Training models...")
    trainer = EarthquakeModelTrainer()
    results = trainer.train_all_models(
        X_train_balanced, y_train_balanced, X_test, y_test
    )
    
    print("\n" + "=" * 60)
    print("Model Comparison Summary (Sorted by Performance)")
    print("=" * 60)
    
    comparison_df = trainer.compare_models()
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n[4/6] Identifying best model...")
    best_name, best_score = trainer.find_best_model()
    
    print(f"\n[5/6] Optimizing {best_name}...")
    best_model = trainer.optimize_best_model(X_train_balanced, y_train_balanced)
    
    print("\n[6/6] Saving model...")
    model_path = model_dir / "model.pkl"
    trainer.save_model(model_path)
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor location: {model_dir / 'preprocessor.pkl'}")
    
    print("\nFeatures used in training:")
    for idx, feature in enumerate(feature_names, 1):
        print(f"{idx}. {feature}")
    
    print("\n" + "=" * 60)
    
    return trainer, X_test, y_test

if __name__ == "__main__":
    trainer, X_test, y_test = main()
