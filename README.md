# MLflow Lab: Capital Bikeshare Daily Demand Forecasting

## Business Scenario
Capital Bikeshare plans its systemwide workload each evening. Its operations team needs an estimate of tomorrow's total rentals to plan staffing and fleet capacity. An estimate that is too low could leave the team unprepared for a busy day, while an estimate that is too high could lead to unnecessary operating costs.

Because forecasts are generated the previous evening, the team relies strictly on information available at that time using time-series lag features. All models are evaluated against a simple planning benchmark: **using yesterday's total rentals as tomorrow's forecast**.

To maintain full traceability across different feature settings, hyperparameters, and data periods, this project implements an MLOps pipeline using **MLflow** for experiment tracking, performance evaluation (MAE, RMSE, MAPE), and Model Registry versioning.

---

## Data Ingestion  & Traceability
This project uses a **local CSV dataset (`data/day.csv`)** containing historical daily bike rental records from the UCI Bike Sharing Dataset.

### Dataset Information

| Attribute | Description |
| --- | --- |
| Dataset | Bike Sharing Dataset |
| Source | UCI Machine Learning Repository |
| File | `data/day.csv` |
| Observations | 730 daily records |
| Data Frequency | Daily |
| Target Variable | `cnt` — Total daily bike rentals |
| Ingestion Method | Local CSV |

The dataset is stored directly in the project's `data/` directory and loaded using Pandas.
---

## Project Structure
```text
capital-bikeshare-mlflow/
│
├── data/
│   └── day.csv           # Local dataset
├── src/
│   └── train_pipeline.py # Unified interactive training & MLflow logging script
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## Prerequisites

- Python and `pip` installed
- A terminal or command prompt
- The project directory (`capital-bikeshare-mlflow/`)
- For the local CSV option: `data/day.csv`

## Getting Started

### 1. Open the project

Open a terminal or command prompt in the project's root directory:

```bash
cd capital-bikeshare-mlflow
```

### 2. Install dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- `mlflow` — Experiment tracking and Model Registry
- `scikit-learn` — Machine learning models and evaluation
- `pandas` — Dataset loading and preprocessing
- `numpy` — Numerical computations


### 3. Run the training pipeline

Execute the main training script:

```bash
python src/train_pipeline.py
```

The pipeline automatically:

1. Loads the historical daily bike rental dataset from `data/day.csv`.
2. Performs time-series feature engineering using **Lag 1 features (yesterday's values)**.
3. Splits the dataset chronologically into **70% training** and **30% testing**.
4. Trains and evaluates three forecasting approaches:
   - **Benchmark:** Yesterday's total rentals as tomorrow's forecast.
   - **Linear Regression:** A regression model using historical features.
   - **Random Forest Regressor:** An ensemble learning model capturing nonlinear relationships.
5. Calculates evaluation metrics for model comparison.
6. Logs experiment parameters, evaluation metrics, and model versions to MLflow using the local tracking database (`mlflow.db`).



**Evaluation metrics:**

| Metric | Description |
| --- | --- |
| **MAE** | Mean Absolute Error — Average absolute difference between predicted and actual rentals. |
| **RMSE** | Root Mean Squared Error — Penalizes larger prediction errors more heavily. |
| **MAPE** | Mean Absolute Percentage Error — Expresses prediction errors as a percentage of actual rentals. |


Lower values indicate better forecasting performance across all three metrics.

The benchmark serves as the baseline for determining whether machine learning models provide meaningful improvements in daily demand forecasting.


### 4. Launch the MLflow UI

Start the MLflow dashboard:

```bash
python -m mlflow ui
```

Open the following URL in your browser:

**http://127.0.0.1:5000**

Use the dashboard to:

- Inspect individual experiment runs.
- Compare benchmark and machine learning model performance.
- Review logged parameters and evaluation metrics.
- Examine tracked model artifacts and registered model versions.


## Quick Start

```bash
# Navigate to project directory
cd capital-bikeshare-mlflow

# Install required dependencies
pip install -r requirements.txt

# Run training using data/day.csv
python src/train_pipeline.py

# Launch MLflow dashboard
python -m mlflow ui
```

Then visit **http://127.0.0.1:5000** to review the experiment results.
