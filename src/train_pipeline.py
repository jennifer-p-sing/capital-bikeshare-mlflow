import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 0. Set SQLite database tracking backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Initialize MLflow Experiment
mlflow.set_experiment("capital_bikeshare_workload_forecasting")

# 2. Interactive Prompt to Choose Data Source
print("=== Capital Bikeshare MLOps Pipeline ===")
choice = input("Choose data source -> Type 'csv' for local file or press Enter for live API (ucimlrepo): ").strip().lower()

if choice == 'csv':
    print("Loading dataset from local CSV file (data/day.csv)...")
    df = pd.read_csv('data/day.csv')
    source_tag = "local_csv"
else:
    print("Fetching dataset dynamically from UCI repository via API...")
    from ucimlrepo import fetch_ucirepo
    bike_sharing = fetch_ucirepo(id=275)
    X_raw = bike_sharing.data.features
    y_raw = bike_sharing.data.targets
    df = pd.concat([X_raw, y_raw], axis=1)
    source_tag = "uci_api"

# 3. Data Preparation & Feature Engineering for Time-Series Forecast
df['dteday'] = pd.to_datetime(df['dteday'])
df = df.sort_values('dteday').reset_index(drop=True)

# Target is tomorrow's total rentals ('cnt'). 
df['target'] = df['cnt']
df['lag_cnt'] = df['cnt'].shift(1)
df_clean = df.dropna(subset=['lag_cnt']).copy()

features = ['lag_cnt', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
X = df_clean[features]
y = df_clean['target']

# Chronological Train-Test Split (80% train, 20% test)
train_size = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# APPROACH 0: Simple Planning Benchmark (Yesterday's Rentals)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Benchmark_{source_tag}"):
    benchmark_preds = X_test['lag_cnt']
    bench_mae = mean_absolute_error(y_test, benchmark_preds)
    bench_rmse = np.sqrt(mean_squared_error(y_test, benchmark_preds))
    bench_mape = mean_absolute_percentage_error(y_test, benchmark_preds) * 100
    
    mlflow.log_param("approach", "benchmark")
    mlflow.log_param("data_source", source_tag)
    mlflow.log_metric("MAE", bench_mae)
    mlflow.log_metric("RMSE", bench_rmse)
    mlflow.log_metric("MAPE", bench_mape)
    print(f"[Benchmark] MAE: {bench_mae:.2f} | RMSE: {bench_rmse:.2f}| MAPE: {bench_mape:.2f}%")

# ---------------------------------------------------------
# APPROACH 1: Linear Regression (Parametric Model)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Linear_Regression_{source_tag}"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_mape = mean_absolute_percentage_error(y_test, lr_preds) * 100
    
    mlflow.log_param("approach", "machine_learning")
    mlflow.log_param("algorithm", "Linear Regression")
    mlflow.log_param("data_source", source_tag)
    mlflow.log_metric("MAE", lr_mae)
    mlflow.log_metric("RMSE", lr_rmse)
    mlflow.log_metric("MAPE", lr_mape)
    
    mlflow.sklearn.log_model(
        lr_model, 
        name="model", 
        registered_model_name="BikeShare_LinearRegression"
    )
    print(f"[Linear Regression] MAE: {lr_mae:.2f} | RMSE: {lr_rmse:.2f}| MAPE: {lr_mape:.2f}%")

# ---------------------------------------------------------
# APPROACH 2: Random Forest Regressor (Non-Linear Ensemble)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Random_Forest_{source_tag}"):
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mape = mean_absolute_percentage_error(y_test, rf_preds) * 100
    
    mlflow.log_params(rf_params)
    mlflow.log_param("approach", "machine_learning")
    mlflow.log_param("algorithm", "Random Forest Regressor")
    mlflow.log_param("data_source", source_tag)
    mlflow.log_metric("MAE", rf_mae)
    mlflow.log_metric("RMSE", rf_rmse)
    mlflow.log_metric("MAPE", rf_mape)
    
    mlflow.sklearn.log_model(
        rf_model, 
        name="model", 
        registered_model_name="BikeShare_RandomForest",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print(f"[Random Forest] MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}| MAPE: {rf_mape:.2f}%")

print(f"\nTraining and MLflow logging completed successfully using {source_tag}!")