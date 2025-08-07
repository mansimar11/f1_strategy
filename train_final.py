# train.py (Hyper-Optimized for XGBoost)
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import optuna
from config import DATABASE_URL

# --- Model Imports ---
from xgboost import XGBRegressor

# --- CONFIGURATION ---
TABLE_NAME = 'race_data_enhanced'
TARGET_COLUMN = 'LapTime'
MLFLOW_EXPERIMENT_NAME = "F1 - XGBoost Hyper-Optimization"
N_TRIALS = 50 # Increase trials for a more thorough search

# --- LOAD & PREPARE DATA ---
print(f"Connecting to database and loading '{TABLE_NAME}' table...")
engine = create_engine(DATABASE_URL)
try:
    df = pd.read_sql_table(TABLE_NAME, engine)
    df.columns = [str(col) for col in df.columns]
except ValueError:
    print(f"Error: Table '{TABLE_NAME}' not found. Please run ingest_data.py first.")
    exit()
print(f"Loaded {len(df)} rows of data.")

print("Performing feature engineering...")
df = pd.get_dummies(df, columns=['Compound', 'grip_type'], prefix=['Compound', 'Grip'])
X = df.drop(columns=[TARGET_COLUMN, 'EventName', 'Driver'])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data prepared. Starting XGBoost hyperparameter tuning...")

# --- OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    """Define the hyperparameter search space for XGBoost."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Use histogram method
        'device': 'cuda',      # Use GPU for training (XGBoost >=2.0.0)
        'n_estimators': trial.suggest_int('n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              verbose=False)
    
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# --- RUN OPTUNA STUDY ---
print(f"Running Optuna optimization for {N_TRIALS} trials...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# --- TRAIN FINAL MODEL & LOG RESULTS ---
print("\n--- Optimization complete. Training final model with best parameters... ---")
best_params = study.best_params
print(f"  Best params found: {best_params}")

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run():
    # Log the best parameters found by Optuna
    mlflow.log_params(best_params)
    
    # Add the GPU parameter back in for the final training
    final_model = XGBRegressor(**best_params, tree_method='hist', device='cuda', random_state=42)
    final_model.fit(X_train_scaled, y_train)

    # Evaluate the final model
    predictions = final_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape_percentage", mape)
    
    # --- PRINT FINAL ACCURACY ---
    print("\n--- Final Model Performance ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  ✅ Model Accuracy: {100 - mape:.2f}%")

    # Log the final model to MLflow
    mlflow.sklearn.log_model(final_model, "model")

# --- SAVE ARTIFACTS FOR THE STREAMLIT APP ---
joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("\n✅ Best overall model, scaler, and columns list saved successfully!")