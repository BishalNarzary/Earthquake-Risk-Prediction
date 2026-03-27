# Earthquake Risk Prediction

A machine learning project that predicts earthquake risk levels — **low**, **medium**, or **high** — based on seismic features using classical ML models trained on USGS earthquake data.

----------------------------------------------------------------------------------------------------

## Overview

This project builds a multi-class classification pipeline that:
- Preprocesses and balances raw seismic data using SMOTE
- Trains and compares multiple ML models (Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression)
- Selects and optimizes the best model via GridSearchCV
- Evaluates performance with metrics, confusion matrix, and precision-recall curves
- Supports single, batch, and CSV-based predictions

----------------------------------------------------------------------------------------------------

## Project Structure

```
Earthquake-Risk-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── earthquake_data.csv                          # Original USGS dataset
│   │   └── unseen_data.csv                              # Unseen dataset for predictions
│   └── processed/
│       ├── X_train.csv                                  # Training features (scaled)
│       ├── X_test.csv                                   # Testing features (scaled)
│       ├── y_train.csv                                  # Training labels (encoded)
│       └── y_test.csv                                   # Testing labels (encoded)
│
├── outputs/
│   ├── models/
│   │   ├── model.pkl                                    # Trained ML model
│   │   └── preprocessor.pkl                             # Fitted scaler & label encoder
│   ├── predictions/
│   │   └── results.csv                                  # Prediction outputs
│   └── visualizations/
│       ├── confusion_matrix.png                         # Model confusion matrix
│       ├── feature_importance.png                       # Feature importance plot
│       └── precision_recall_curve.png                   # PR curves for each class
│
├── src/
│   ├── data_preprocessing.py                            # Data cleaning & transformation
│   ├── model_training.py                                # Model training & optimization
│   ├── model_evaluation.py                              # Model performance evaluation
│   └── predict.py                                       # Prediction interface
│
├── notebooks/
│   └── earthquake_risk_prediction_pipeline.ipynb        # Jupyter notebook
│
├── README.md                                            # Project documentation
└── requirements.txt                                     # Python dependencies
```

----------------------------------------------------------------------------------------------------

## Features

The model is trained on the following features:

`latitude`      -  Epicenter latitude (decimal degrees) 
`longitude`     -  Epicenter longitude (decimal degrees) 
`depth`         -  Depth below Earth's surface (km) 
`geo_distance`  -  Euclidean distance from origin — `sqrt(latitude² + longitude²)`

**Target:** `risk` — categorized into `low`, `medium`, `high` based on magnitude quartiles

----------------------------------------------------------------------------------------------------

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/BishalNarzary/Earthquake-Risk-Prediction.git
cd Earthquake-Risk-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess the data
```bash
python src/data_preprocessing.py
```

### 4. Train the model
```bash
python src/model_training.py
```

### 5. Evaluate the model
```bash
python src/model_evaluation.py
```

### 6. Run predictions

**From a CSV file:**
```bash
python src/predict.py --csv data/raw/unseen_data.csv --output outputs/predictions/results.csv
```

**Interactive mode:**
```bash
python src/predict.py --interactive
```

**Example predictions:**
```bash
python src/predict.py --examples
```

----------------------------------------------------------------------------------------------------

## Data

Source: [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/)  
The dataset contains seismic event records including location, depth, magnitude, and station metadata.
