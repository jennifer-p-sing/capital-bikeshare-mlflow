# MLflow Lab: Capital Bikeshare Daily Demand Forecasting

## Business Scenario
Capital Bikeshare plans its systemwide workload each evening. Its operations team needs an estimate of tomorrow's total rentals to plan staffing and fleet capacity. An estimate that is too low could leave the team unprepared for a busy day, while an estimate that is too high could lead to unnecessary operating costs.

Because forecasts are generated the previous evening, the team relies strictly on information available at that time using time-series lag features. All models are evaluated against a simple planning benchmark: **using yesterday's total rentals as tomorrow's forecast**.

To maintain full traceability across different feature settings, hyperparameters, and data periods, this project implements an MLOps pipeline using **MLflow** for experiment tracking, performance evaluation (MAE, RMSE, MAPE), and Model Registry versioning.

---

## Data Ingestion Options & Traceability

This project supports two interchangeable methods for loading the daily dataset, managed via an interactive prompt when executing the training script:

1. **Option A: Live API Ingestion (`ucimlrepo`)**
   - Dynamically fetches the dataset directly from the UCI Machine Learning Repository at runtime.
   - Ideal for automated CI/CD pipelines or clean environments where local data files are not pre-packaged.
   - Samples: 17,378 (as of 15/10/2026)

2. **Option B: Local CSV Ingestion (`data/day.csv`)**
   - Loads the pre-downloaded daily snapshot directly from the local `data/` directory.
   - Acts as a reliable offline fallback option if network or firewall restrictions prevent live API fetching during review.
   - Samples 730

*Note on Tracking:* Regardless of the ingestion method chosen, the resulting data is normalized identically and logged with a specific data source tag (`uci_api` vs. `local_csv`) inside MLflow to maintain full experiment traceability.
---

## Project Structure
```text
capital-bikeshare-mlflow/
│
├── data/
│   └── day.csv           # Local dataset copy (optional fallback)
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

The dependencies include `mlflow`, `scikit-learn`, `ucimlrepo`, `pandas`, and `numpy`.

### 3. Run the training pipeline

Execute the main training script:

```bash
python src/train_pipeline.py
```

When prompted, choose a data source:

| Option | Input | Description |
| --- | --- | --- |
| **A: Live API** (default) | Press **Enter** | Fetches the dataset from the UCI Machine Learning Repository via `ucimlrepo`. |
| **B: Local CSV** | Type `csv` and press **Enter** | Loads the dataset from `data/day.csv`. |

The pipeline automatically:

1. Performs time-series feature engineering using **Lag 1 features a.k.a. Yesterday's value**.
2. Splits the dataset chronologically into **70% training** and **30% testing**.
3. Trains and evaluates the **Benchmark**, **Linear Regression**, and **Random Forest Regressor**.
4. Logs parameters, evaluation metrics, and model versions to the local MLflow tracking database (`mlflow.db`).


**Evaluation metrics:**

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

### 4. Launch the MLflow UI

Start the MLflow dashboard:

```bash
python -m mlflow ui
```

Open the following URL in your browser:

**http://127.0.0.1:5000**

Use the dashboard to inspect experiment runs, compare model metrics, and verify registered models.

## Quick Start

```bash
cd capital-bikeshare-mlflow
pip install -r requirements.txt
python src/train_pipeline.py
# Choose Enter for the live API or type csv for the local dataset.
python -m mlflow ui
```

Then visit **http://127.0.0.1:5000**.
