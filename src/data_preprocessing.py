import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import os

class EarthquakePreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def _remove_outliers(self, df):
        Q1 = df[['latitude', 'longitude', 'depth', 'mag']].quantile(0.25)
        Q3 = df[['latitude', 'longitude', 'depth', 'mag']].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[['latitude', 'longitude', 'depth', 'mag']] < (Q1 - 1.5 * IQR)) |
                  (df[['latitude', 'longitude', 'depth', 'mag']] > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df

    def _feature_engineering(self, df):
        df['geo_distance'] = np.sqrt(df['latitude']**2 + df['longitude']**2)
        return df

    def _categorize_magnitude(self, df):
        df['risk'] = pd.qcut(df['mag'], q=3, labels=['low', 'medium', 'high'])
        return df

    def fit(self, df):
        df = df[['latitude', 'longitude', 'depth', 'mag']].dropna()
        df = self._remove_outliers(df)
        df = self._feature_engineering(df)
        df = self._categorize_magnitude(df)

        X = df[['latitude', 'longitude', 'depth', 'geo_distance']]
        y = df['risk']

        self.scaler.fit(X)
        self.label_encoder.fit(y)

        return self

    def transform(self, df):
        df = df[['latitude', 'longitude', 'depth', 'mag']].dropna()
        df = self._remove_outliers(df)
        df = self._feature_engineering(df)
        df = self._categorize_magnitude(df)

        X = df[['latitude', 'longitude', 'depth', 'geo_distance']]
        y = df['risk']

        X_scaled = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y)

        return X_scaled, y_encoded

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def split(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

    def get_class_mapping(self):
        return {label: idx for idx, label in enumerate(self.label_encoder.classes_)}

def main():
    print("=" * 50)
    print("Earthquake Risk Prediction - Data Preprocessing")
    print("=" * 50)
    
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "earthquake_data.csv"
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    print("\n[1/4] Loading raw data...")
    df = pd.read_csv(raw_path)

    print("\n[2/4] Preprocessing data...")
    preprocessor = EarthquakePreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split(X, y)

    print("\n[3/4] Saving processed files...")
    pd.DataFrame(X_train).to_csv(processed_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["risk"]).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["risk"]).to_csv(processed_dir / "y_test.csv", index=False)
    
    print("\n[4/4] Saving preprocessor...")
    model_dir = project_root / "outputs" / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    preprocessor_path = model_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\nPreprocessor saved to: {preprocessor_path}")
    print("Preprocessing complete!")
    print("Class Mapping:", preprocessor.get_class_mapping())

if __name__ == "__main__":
    main()
