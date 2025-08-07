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
from xgboost import XGBRegressor

# CONFIGURATION
TABLE_NAME = 'race_data_enhanced'
TARGET_COLUMN = 'LapTime'
MLFLOW_EXPERIMENT_NAME = "F1 - XGBoost Advanced Features"
N_TRIALS = 30

# LOAD DATA
print(f"Connecting to database and loading '{TABLE_NAME}' table...")
engine = create_engine(DATABASE_URL)
try:
    df = pd.read_sql_table(TABLE_NAME, engine)
    df.columns = [str(col) for col in df.columns]
except ValueError:
    print(f"Error: Table '{TABLE_NAME}' not found. Please run ingest_data.py first.")
    exit()
print(f"Loaded {len(df)} rows of data.")

print("Performing advanced feature engineering...")

# Lag and rolling features
df = df.sort_values(by=['Year', 'EventName', 'Driver', 'LapNumber'])
df['PreviousLapTime'] = df.groupby(['Year', 'EventName', 'Driver'])['LapTime'].shift(1)
df['Rolling3LapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['LapTime'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
df['Rolling3LapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['Rolling3LapAvg'].shift(1)
df['Rolling5LapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['LapTime'].rolling(window=5, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
df['Rolling5LapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['Rolling5LapAvg'].shift(1)
df['ExpLapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['LapTime'].apply(lambda x: x.ewm(span=5, min_periods=1).mean()).reset_index(level=[0,1,2], drop=True)
df['ExpLapAvg'] = df.groupby(['Year', 'EventName', 'Driver'])['ExpLapAvg'].shift(1)

df['DriverTrackAvg'] = df.groupby(['EventName', 'Driver'])['LapTime'].transform('mean')
if 'Team' in df.columns:
    df['TeamTrackAvg'] = df.groupby(['EventName', 'Team'])['LapTime'].transform('mean')
df['TrackAvg'] = df.groupby(['EventName'])['LapTime'].transform('mean')

# Drop first laps/NaNs
df.dropna(inplace=True)
print(f"Feature engineering complete. {len(df)} rows remaining for training.")

# Dummy coding
df = pd.get_dummies(df, columns=['Compound', 'grip_type'], prefix=['Compound', 'Grip'])
drop_cols = [col for col in [TARGET_COLUMN, 'EventName', 'Driver', 'Team'] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data prepared. Starting XGBoost hyperparameter tuning...")

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 400, 2400),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.24, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 17),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

print(f"Running Optuna optimization for {N_TRIALS} trials...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n--- Optimization complete. Training final model with best parameters... ---")
best_params = study.best_params
print(f"  Best params found: {best_params}")

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_params(best_params)
    final_model = XGBRegressor(**best_params, tree_method='hist', random_state=42)
    final_model.fit(X_train_scaled, y_train)
    predictions = final_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape_percentage", mape)
    print("\n--- Final Model Performance ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  ✅ Model Accuracy: {100 - mape:.2f}%")
    mlflow.sklearn.log_model(final_model, "model")

final_columns = [col for col in X.columns if col != 'Team']
joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(final_columns, 'model_columns.pkl')
print("\n✅ Best overall model, scaler, and columns list saved successfully!")
