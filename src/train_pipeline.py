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

# 2. # 2. Load Dataset from Local CSV
print("=== Capital Bikeshare MLOps Pipeline ===")
print("Loading dataset from local CSV file (data/day.csv)...")
df = pd.read_csv('data/day.csv')
source_tag = "local_csv" # I initially coded it using 2 sources: api and local csv. I removed the api (bec it was hourly) so that i do not have to adjust the whole code I just kept this source tag

# 3. Data Preparation & Feature Engineering for Time-Series Forecast
df['dteday'] = pd.to_datetime(df['dteday'])
df = df.sort_values('dteday').reset_index(drop=True)

# Target is tomorrow's total rentals ('cnt'). 
df['target'] = df['cnt']
df['lag_cnt'] = df['cnt'].shift(1) #containing yesterday's total rentals for each row coz i already ordered it by date. This is the benchmark model for tomorrow's rentals prediction.
df_clean = df.dropna(subset=['lag_cnt']).copy()

features = ['lag_cnt', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
X = df_clean[features]
y = df_clean['target']

# Chronological Train-Test Split (70% train, 30% test)
train_size = int(len(df_clean) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# APPROACH 0: Simple Planning Benchmark (Yesterday's Rentals)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Benchmark_{source_tag}"):
    # Train predictions & metrics
    bench_train_preds = X_train['lag_cnt']
    bench_train_mae = mean_absolute_error(y_train, bench_train_preds)
    bench_train_rmse = np.sqrt(mean_squared_error(y_train, bench_train_preds))
    bench_train_mape = mean_absolute_percentage_error(y_train, bench_train_preds) * 100

    # Test predictions & metrics
    bench_test_preds = X_test['lag_cnt']
    bench_test_mae = mean_absolute_error(y_test, bench_test_preds)
    bench_test_rmse = np.sqrt(mean_squared_error(y_test, bench_test_preds))
    bench_test_mape = mean_absolute_percentage_error(y_test, bench_test_preds) * 100
    
    # Log parameters
    mlflow.log_param("approach", "benchmark")
    mlflow.log_param("data_source", source_tag)
    
    # Log training metrics
    mlflow.log_metric("train_mae", bench_train_mae)
    mlflow.log_metric("train_rmse", bench_train_rmse)
    mlflow.log_metric("train_mape", bench_train_mape)
    
    # Log testing metrics
    mlflow.log_metric("test_mae", bench_test_mae)
    mlflow.log_metric("test_rmse", bench_test_rmse)
    mlflow.log_metric("test_mape", bench_test_mape)
    
    print(f"[Benchmark] Train MAE: {bench_train_mae:.2f} | Test MAE: {bench_test_mae:.2f}")
    print(f"[Benchmark] Train RMSE: {bench_train_rmse:.2f} | Test RMSE: {bench_test_rmse:.2f}")
    print(f"[Benchmark] Train MAPE: {bench_train_mape:.2f}% | Test MAPE: {bench_test_mape:.2f}%")   

# ---------------------------------------------------------
# APPROACH 1: Linear Regression (Parametric Model)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Linear_Regression_{source_tag}"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Train predictions & metrics
    lr_train_preds = lr_model.predict(X_train)
    lr_train_mae = mean_absolute_error(y_train, lr_train_preds)
    lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_preds))
    lr_train_mape = mean_absolute_percentage_error(y_train, lr_train_preds) * 100

    # Test predictions & metrics
    lr_test_preds = lr_model.predict(X_test)
    lr_test_mae = mean_absolute_error(y_test, lr_test_preds)
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_preds))
    lr_test_mape = mean_absolute_percentage_error(y_test, lr_test_preds) * 100
    
    # Log parameters
    mlflow.log_param("approach", "machine_learning")
    mlflow.log_param("algorithm", "Linear Regression")
    mlflow.log_param("data_source", source_tag)
    
    # Log training metrics
    mlflow.log_metric("train_mae", lr_train_mae)
    mlflow.log_metric("train_rmse", lr_train_rmse)
    mlflow.log_metric("train_mape", lr_train_mape)
    
    # Log testing metrics
    mlflow.log_metric("test_mae", lr_test_mae)
    mlflow.log_metric("test_rmse", lr_test_rmse)
    mlflow.log_metric("test_mape", lr_test_mape)
    
    mlflow.sklearn.log_model(
        lr_model, 
        name="model", 
        registered_model_name="BikeShare_LinearRegression"
    )
    print(f"[Linear Regression] Train MAE: {lr_train_mae:.2f} | Test MAE: {lr_test_mae:.2f}")
    print(f"[Linear Regression] Train RMSE: {lr_train_rmse:.2f} | Test RMSE: {lr_test_rmse:.2f}")
    print(f"[Linear Regression] Train MAPE: {lr_train_mape:.2f}% | Test MAPE: {lr_test_mape:.2f}%")

# ---------------------------------------------------------
# APPROACH 2: Random Forest Regressor (Non-Linear Ensemble)
# ---------------------------------------------------------
with mlflow.start_run(run_name=f"Random_Forest_{source_tag}"):
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    
    # Train predictions & metrics
    rf_train_preds = rf_model.predict(X_train)
    rf_train_mae = mean_absolute_error(y_train, rf_train_preds)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_preds))
    rf_train_mape = mean_absolute_percentage_error(y_train, rf_train_preds) * 100

    # Test predictions & metrics
    rf_test_preds = rf_model.predict(X_test)
    rf_test_mae = mean_absolute_error(y_test, rf_test_preds)
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_preds))
    rf_test_mape = mean_absolute_percentage_error(y_test, rf_test_preds) * 100
    
    # Log parameters
    mlflow.log_params(rf_params)
    mlflow.log_param("approach", "machine_learning")
    mlflow.log_param("algorithm", "Random Forest Regressor")
    mlflow.log_param("data_source", source_tag)
    
    # Log training metrics
    mlflow.log_metric("train_mae", rf_train_mae)
    mlflow.log_metric("train_rmse", rf_train_rmse)
    mlflow.log_metric("train_mape", rf_train_mape)
    
    # Log testing metrics
    mlflow.log_metric("test_mae", rf_test_mae)
    mlflow.log_metric("test_rmse", rf_test_rmse)
    mlflow.log_metric("test_mape", rf_test_mape)
    
    mlflow.sklearn.log_model(
        rf_model, 
        name="model", 
        registered_model_name="BikeShare_RandomForest",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print(f"[Random Forest] Train MAE: {rf_train_mae:.2f} | Test MAE: {rf_test_mae:.2f}")
    print(f"[Random Forest] Train RMSE: {rf_train_rmse:.2f} | Test RMSE: {rf_test_rmse:.2f}")
    print(f"[Random Forest] Train MAPE: {rf_train_mape:.2f}% | Test MAPE: {rf_test_mape:.2f}%")

print(f"\nTraining and MLflow logging (Train + Test metrics) completed successfully using {source_tag}!")