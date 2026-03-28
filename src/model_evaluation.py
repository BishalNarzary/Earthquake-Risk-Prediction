import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import joblib
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from data_preprocessing import EarthquakePreprocessor
except ImportError:
    import importlib.util
    
    preprocessing_path = Path(__file__).resolve().parent / "data_preprocessing.py"
    spec = importlib.util.spec_from_file_location("data_preprocessing", preprocessing_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load data_preprocessing module from {preprocessing_path}")
    
    data_preprocessing = importlib.util.module_from_spec(spec)
    sys.modules["data_preprocessing"] = data_preprocessing
    spec.loader.exec_module(data_preprocessing)
    EarthquakePreprocessor = data_preprocessing.EarthquakePreprocessor

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EarthquakeModelEvaluator:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        report = classification_report(
            y_test, y_pred,
            target_names=self.labels,
            output_dict=True
        )

        return metrics, y_pred, report

    def cross_validate(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=5, scoring='f1_weighted')

        print("\nCross Validation (F1-weighted):")
        print(f"Scores: {scores}")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")

    def plot_confusion_matrix(self, y_test, y_pred, save_path):
        cm = confusion_matrix(y_test, y_pred)
 
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=self.labels,
                    yticklabels=self.labels)
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved to: {save_path}")

    def plot_feature_importance(self, model, feature_names, save_path):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
 
            plt.figure(figsize=(10, 6))
            viridis = plt.colormaps.get_cmap('viridis')
            colors = viridis(importances[indices] / importances[indices].max())
            plt.barh(range(len(indices)), importances[indices], color=colors)
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.gca().invert_yaxis()
            plt.title("Feature Importance", fontsize=14, fontweight='bold')
            plt.xlabel("Importance", fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nFeature importance plot saved to: {save_path}")
        else:
            print("\nModel does not have feature_importances_ attribute. Skipping feature importance plot.")

    def plot_precision_recall(self, X_test, y_test, save_path):
        if not hasattr(self.model, "predict_proba"):
            print("\nModel does not support predict_proba. Skipping PR curve.")
            return
 
        X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
        y_score = self.model.predict_proba(X_test_array)
        y_test_bin = label_binarize(y_test, classes=range(len(self.labels)))
        y_test_bin = np.asarray(y_test_bin)

        plt.figure(figsize=(8, 6))
 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, label in enumerate(self.labels):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, label=label, linewidth=2.5, color=colors[i])
 
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nPrecision-recall curve saved to: {save_path}")

def display_metrics(metrics, class_report, labels):
    print("\n" + "=" * 60)
    print("Overall Model Performance")
    print("=" * 60)
 
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
 
    print("\n" + "=" * 60)
    print("Per-Class Performance")
    print("=" * 60)
 
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 60)
 
    for label in labels:
        stats = class_report[label]
        print(f"{label:<15} {stats['precision']:<12.4f} {stats['recall']:<12.4f} "
              f"{stats['f1-score']:<12.4f} {int(stats['support'])}")

def compare_models(model_path, X_test, y_test):
    loaded = joblib.load(model_path)
 
    if not isinstance(loaded, dict):
        loaded = {"\nSavedModel": loaded}
 
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
 
    for name, model in loaded.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name}: F1 = {f1:.4f}")

def main():
    print("=" * 60)
    print("Earthquake Risk Prediction - Model Evaluation")
    print("=" * 60)
 
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "outputs" / "models" / "model.pkl"
    preprocessor_path = project_root / "outputs" / "models" / "preprocessor.pkl"
    vis_dir = project_root / "outputs" / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
 
    print("\n[1/6] Loading processed data...")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["risk"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["risk"]
 
    feature_names = ['latitude', 'longitude', 'depth', 'geo_distance']
    X_train.columns = feature_names
    X_test.columns = feature_names
 
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Features: {', '.join(feature_names)}")
 
    print("\n[2/6] Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    labels = preprocessor.label_encoder.classes_
    print(f"  Risk levels: {', '.join(labels)}")
 
    evaluator = EarthquakeModelEvaluator(model, labels)
 
    print("\n[3/6] Evaluating model on test set...")
    X_test_array = X_test.values
    X_train_array = X_train.values
    metrics, y_pred, report = evaluator.evaluate(X_test_array, y_test)
 
    display_metrics(metrics, report, labels)
 
    print("\n[4/6] Running cross-validation on training set...")
    evaluator.cross_validate(X_train, y_train)
 
    print("\n[5/6] Generating visualizations...")
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        vis_dir / "confusion_matrix.png"
    )
    evaluator.plot_feature_importance(
        model, feature_names,
        vis_dir / "feature_importance.png"
    )
    evaluator.plot_precision_recall(
        X_test, y_test,
        vis_dir / "precision_recall_curve.png"
    )
 
    print("\n[6/6] Comparing models...")
    compare_models(model_path, X_test_array, y_test)
 
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nVisualization files saved in: {vis_dir}")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - precision_recall_curve.png")
    
    print(f"\nModel Performance Summary:")
    print(f"  Best F1-Score: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
 
if __name__ == "__main__":
    main()
