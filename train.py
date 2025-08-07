<<<<<<< HEAD
# train.py (Upgraded with Hyperparameter Tuning)
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import optuna
import numpy as np
from config import DATABASE_URL

# --- Model Imports ---
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- CONFIGURATION ---
TABLE_NAME = 'race_data_enhanced'
TARGET_COLUMN = 'LapTime'
MLFLOW_EXPERIMENT_NAME = "F1 Lap Time - Tuned Models"
N_TRIALS_PER_MODEL = 25 # Number of tuning trials for each model

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
print("Data prepared. Starting hyperparameter tuning for top models...")

# --- MLFLOW EXPERIMENT ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- HYPERPARAMETER TUNING AND TRAINING LOOP ---
top_models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]

for model_name in top_models:
    print(f"\n--- Tuning {model_name} ---")
    
    def objective(trial):
        # Define the search space for hyperparameters for each model
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42, 'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'random_state': 42,
                'tree_method': 'gpu_hist'  # Use GPU if available
            }
            model = XGBRegressor(**params)
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'random_state': 42, 'n_jobs': -1
            }
            model = LGBMRegressor(**params)
        elif model_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 12),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'random_state': 42, 'verbose': 0
            }
            model = CatBoostRegressor(**params)

        # Train with early stopping where applicable
        fit_params = {}
        if model_name == "LightGBM":
            from lightgbm.callback import early_stopping
            fit_params = {'eval_set': [(X_test_scaled, y_test)], 'callbacks': [early_stopping(50, verbose=False)]}
        elif model_name == "CatBoost":
            fit_params = {'eval_set': [(X_test_scaled, y_test)], 'early_stopping_rounds': 50}
        # For XGBoost, only pass eval_set (no early_stopping_rounds for older versions)
        elif model_name == "XGBoost":
            fit_params = {'eval_set': [(X_test_scaled, y_test)]}

        model.fit(X_train_scaled, y_train, **fit_params)
        
        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    # Run the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS_PER_MODEL, show_progress_bar=True)
    
    # --- Log results of the best trial to MLflow ---
    with mlflow.start_run(run_name=f"{model_name}_Tuned"):
        best_params = study.best_params
        print(f"  Best params for {model_name}: {best_params}")
        
        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_name)
        
        # Train final model with the best parameters
        if model_name == "RandomForest":
            final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        if model_name == "XGBoost":
            final_model = XGBRegressor(**best_params, random_state=42, tree_method='gpu_hist')
        if model_name == "LightGBM":
            final_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
        if model_name == "CatBoost":
            final_model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
            
        final_model.fit(X_train_scaled, y_train)
        
        predictions = final_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape_percentage", mape)

        print(f"  --> Tuned {model_name} RMSE: {rmse:.4f}")
        print(f"  --> Tuned {model_name} R2 Score: {r2:.4f}")
        print(f"  --> Tuned {model_name} MAPE: {mape:.2f}%")
        
        mlflow.sklearn.log_model(final_model, "model")

print("\n--- All models tuned. Finding the best overall model... ---")

# --- SAVE THE ABSOLUTE BEST MODEL ---
best_run = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT_NAME],
    order_by=["metrics.rmse ASC"],
    max_results=1
).iloc[0]

best_run_id = best_run["run_id"]
best_model_rmse = best_run["metrics.rmse"]
best_model_name = best_run["params.model_type"]

print(f"🏆 Overall Best Model: {best_model_name} with Tuned RMSE: {best_model_rmse:.4f}")

best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("✅ Best overall model saved successfully!")
=======
# train.py (Simple & Reliable XGBoost Version with model.pkl saving)
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
from config import DATABASE_URL
from track_config import TRACK_CHARACTERISTICS

print("--- Simple & Reliable Scientific Model Training ---")

EXPERIMENT_NAME = "F1 Reliable XGBoost Model"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 1. Load and Prepare Data ---
engine = create_engine(DATABASE_URL)
df = pd.read_sql('SELECT "LapTime", "LapNumber", "TyreLife", "Compound", "EventName" FROM laps', engine)
track_df = pd.DataFrame.from_dict(TRACK_CHARACTERISTICS, orient='index')
df = df.join(track_df, on='EventName')
df.dropna(inplace=True)
df = df[df['LapTime'].between(60, 150)]

# --- 2. Feature Engineering ---
df['fuel_correction'] = (df['laps'] - df['LapNumber']) * 0.03
df['track_evolution'] = (df['LapNumber'] - 1) * -0.015
df['tyre_load'] = df['TyreLife'] * df['downforce']

y = df['LapTime']
X = df[['TyreLife', 'Compound', 'track_temp', 'abrasiveness', 'downforce',
        'fuel_correction', 'track_evolution', 'tyre_load', 'grip_type']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Preprocessing ---
numeric_features = [col for col in X.columns if X[col].dtype != 'object']
categorical_features = ['Compound', 'grip_type']
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# --- 4. Train Model ---
with mlflow.start_run(run_name="XGBoost_Reliable") as run:
    model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42, n_jobs=-1)
    pipeline = make_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(y_test, predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.sklearn.log_model(pipeline, "model")
    mlflow.set_tag("best_overall_model", "true")

    # --- Save for Streamlit app to use ---
    joblib.dump(pipeline, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print(f"✅ XGBoost Model logged. RMSE: {rmse:.4f}s | MAPE: {mape:.2%}")
>>>>>>> 6654cc450a78f302f5fe365ff121aa506498f2a0
