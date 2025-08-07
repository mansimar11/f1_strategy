# create_defaults_summary.py
import pandas as pd
from sqlalchemy import create_engine
import json
from config import DATABASE_URL

TABLE_NAME = 'race_data_enhanced'
OUTPUT_FILE = 'track_defaults.json'

print(f"Loading data from '{TABLE_NAME}' to generate default values...")
engine = create_engine(DATABASE_URL)
try:
    df = pd.read_sql_table(TABLE_NAME, engine)
    df.columns = [str(col) for col in df.columns]
except ValueError:
    print(f"Error: Table '{TABLE_NAME}' not found. Please run ingest_data.py first.")
    exit()

default_columns = ['track_temp', 'downforce', 'abrasiveness']
defaults_df = df.groupby('EventName')[default_columns].first()
defaults_dict = defaults_df.to_dict(orient='index')

with open(OUTPUT_FILE, 'w') as f:
    json.dump(defaults_dict, f, indent=4)

print(f"✅ Successfully created '{OUTPUT_FILE}' with historical defaults.")