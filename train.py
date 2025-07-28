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
