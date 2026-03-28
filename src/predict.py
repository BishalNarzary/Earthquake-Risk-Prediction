import numpy as np
import pandas as pd
import argparse
import joblib
import sys
from pathlib import Path
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

class EarthquakeRiskPredictor:
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.risk_labels = self.preprocessor.label_encoder.classes_
        
    def predict(self, features):
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame([features], columns=['latitude', 'longitude', 'depth', 'mag'])
        
        df = df.copy()
        df = self._preprocess_features(df)
        
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        risk_level = self.risk_labels[prediction]
        
        prob_dict = {
            label: prob for label, prob in zip(self.risk_labels, probabilities)
        }
        
        return risk_level, prob_dict
    
    def _preprocess_features(self, df):
        df = df.copy()
        df['geo_distance'] = np.sqrt(df['latitude']**2 + df['longitude']**2)
        
        X = df[['latitude', 'longitude', 'depth', 'geo_distance']]
        X_scaled = self.preprocessor.scaler.transform(X)
        
        return X_scaled
    
    def predict_batch(self, features_df):
        X_scaled = self._preprocess_features(features_df)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            risk_level = self.risk_labels[pred]
            prob_dict = {
                label: prob for label, prob in zip(self.risk_labels, probs)
            }
            results.append({
                'prediction': risk_level,
                'probabilities': prob_dict
            })
        
        return results
    
    def predict_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        
        required_cols = ['latitude', 'longitude', 'depth', 'mag']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        results = self.predict_batch(df[required_cols])
        
        df['predicted_risk'] = [r['prediction'] for r in results]
        df['confidence'] = [max(r['probabilities'].values()) for r in results]
        
        return df, results

def example_predictions():
    print("=" * 60)
    print("Earthquake Risk Prediction - Examples")
    print("=" * 60)
    
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "outputs" / "models" / "model.pkl"
    preprocessor_path = project_root / "outputs" / "models" / "preprocessor.pkl"
    
    print("\n<> Loading trained model and preprocessor...")
    predictor = EarthquakeRiskPredictor(model_path, preprocessor_path)
    print("  Model and preprocessor loaded successfully")
    print(f"  Risk levels: {predictor.risk_labels}")
    
    print("\n" + "-" * 60)
    print("Example 1: Low-Risk Earthquake (Small magnitude, Shallow)")
    print("-" * 60)
    low_risk_earthquake = {
        'latitude': 34.05,
        'longitude': -118.25,
        'depth': 5.0,
        'mag': 3.2
    }
    
    print(f"Input: {low_risk_earthquake}")
    risk, probs = predictor.predict(low_risk_earthquake)
    print(f"\nPredicted Risk Level: {risk.upper()}")
    print("Probabilities:")
    for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label.capitalize():<15}: {prob:.2%} {'█' * int(prob * 50)}")
    
    print("\n" + "-" * 60)
    print("Example 2: Medium-Risk Earthquake (Moderate magnitude)")
    print("-" * 60)
    medium_risk_earthquake = {
        'latitude': 35.68,
        'longitude': 139.65,
        'depth': 35.0,
        'mag': 5.5
    }
    
    print(f"Input: {medium_risk_earthquake}")
    risk, probs = predictor.predict(medium_risk_earthquake)
    print(f"\nPredicted Risk Level: {risk.upper()}")
    print("Probabilities:")
    for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label.capitalize():<15}: {prob:.2%} {'█' * int(prob * 50)}")
    
    print("\n" + "-" * 60)
    print("Example 3: High-Risk Earthquake (Large magnitude, Deep)")
    print("-" * 60)
    high_risk_earthquake = {
        'latitude': 38.30,
        'longitude': 142.37,
        'depth': 60.0,
        'mag': 7.8
    }
    
    print(f"Input: {high_risk_earthquake}")
    risk, probs = predictor.predict(high_risk_earthquake)
    print(f"\nPredicted Risk Level: {risk.upper()}")
    print("Probabilities:")
    for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label.capitalize():<15}: {prob:.2%} {'█' * int(prob * 50)}")
    
    print("\n" + "-" * 60)
    print("Example 4: Shallow Earthquake (Potentially damaging)")
    print("-" * 60)
    shallow_earthquake = {
        'latitude': 37.77,
        'longitude': -122.42,
        'depth': 2.0,
        'mag': 4.5
    }
    
    print(f"Input: {shallow_earthquake}")
    risk, probs = predictor.predict(shallow_earthquake)
    print(f"\nPredicted Risk Level: {risk.upper()}")
    print("Probabilities:")
    for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label.capitalize():<15}: {prob:.2%} {'█' * int(prob * 50)}")
    
    print("\n" + "-" * 60)
    print("Example 5: Batch Prediction (Multiple Earthquakes)")
    print("-" * 60)
    
    batch_data = pd.DataFrame([
        {'latitude': 34.05, 'longitude': -118.25, 'depth': 10.0, 'mag': 4.0},
        {'latitude': 35.68, 'longitude': 139.65, 'depth': 50.0, 'mag': 6.2},
        {'latitude': 40.73, 'longitude': -74.00, 'depth': 8.0, 'mag': 3.5},
    ])
    
    print("\nInput DataFrame:")
    print(batch_data.to_string(index=False))
    
    results = predictor.predict_batch(batch_data)
    
    print("\nPredictions:")
    for i, result in enumerate(results, 1):
        print(f"\n  Earthquake {i}: {result['prediction'].upper()}")
        for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {label.capitalize():<10}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("Example Predictions Complete!")
    print("=" * 60)

def interactive_mode():
    print("=" * 60)
    print("Earthquake Risk Prediction - Interactive Mode")
    print("=" * 60)
    
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "outputs" / "models" / "model.pkl"
    preprocessor_path = project_root / "outputs" / "models" / "preprocessor.pkl"
    
    predictor = EarthquakeRiskPredictor(model_path, preprocessor_path)
    
    print("\nEnter earthquake parameters (or 'q' to quit):\n")
    
    while True:
        try:
            lat = input("Latitude: ")
            if lat.lower() == 'q':
                break
            
            lon = input("Longitude: ")
            depth = input("Depth (km): ")
            mag = input("Magnitude: ")
            
            features = {
                'latitude': float(lat),
                'longitude': float(lon),
                'depth': float(depth),
                'mag': float(mag)
            }
            
            risk, probs = predictor.predict(features)
            
            print("\n" + "-" * 40)
            print(f"Predicted Risk Level: {risk.upper()}")
            print("\nProbabilities:")
            for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label.capitalize():<15}: {prob:.2%}")
            print("-" * 40 + "\n")
        except ValueError as e:
            print(f"Error: Invalid input. Please enter numeric values.\n")
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(
        description='Predict earthquake risk level from seismic features'
    )
    parser.add_argument('--examples', action='store_true', help='Run example predictions')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--csv', type=str, help='Path to CSV file with earthquake data')
    parser.add_argument('--output', type=str, help='Output path for predictions (used with --csv)')
    
    args = parser.parse_args()
    
    if args.examples:
        example_predictions()
    elif args.interactive:
        interactive_mode()
    elif args.csv:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "outputs" / "models" / "model.pkl"
        preprocessor_path = project_root / "outputs" / "models" / "preprocessor.pkl"
        
        predictor = EarthquakeRiskPredictor(model_path, preprocessor_path)
        
        print(f"\nLoading earthquake data from {args.csv}...")
        df, results = predictor.predict_from_csv(args.csv)
        
        if args.output:
            os.makedirs(Path(args.output).parent, exist_ok=True)
            df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print("\nPredictions:")
            print(df.to_string(index=False))
    else:
        print("Usage:")
        print("  python predict.py --examples          # Run example predictions")
        print("  python predict.py --interactive       # Interactive mode")
        print("  python predict.py --csv <file>        # Predict from CSV")
        print("  python predict.py --csv <file> --output <out.csv>")

if __name__ == "__main__":
    main()
