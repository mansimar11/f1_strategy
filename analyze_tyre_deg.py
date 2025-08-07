import pandas as pd
from sqlalchemy import create_engine
import json
from config import DATABASE_URL

TABLE_NAME = 'race_data_enhanced'
OUTPUT_FILE = 'tyre_deg_curves.json'

print(f"Loading data from '{TABLE_NAME}' to analyze tyre degradation...")
engine = create_engine(DATABASE_URL)
try:
    df = pd.read_sql_table(TABLE_NAME, engine)
    df.columns = [str(col) for col in df.columns]
except ValueError:
    print(f"Error: Table '{TABLE_NAME}' not found. Please run ingest_data.py first.")
    exit()

df = df.sort_values(by=['Year', 'EventName', 'Driver', 'Stint', 'LapNumber'])

df['BaselineLapTime'] = df.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapTime'].transform(
    lambda x: x.iloc[1:4].mean()
)
df['LapTimeDelta'] = df['LapTime'] - df['BaselineLapTime']

# Only use realistic racing laps (no major outliers, not first lap of stint)
df_filtered = df[(df['LapTimeDelta'] < 10) & (df['LapTimeDelta'] > -2) & (df['TyreLife'] > 1)].copy()

# Calculate mean degradation per event+compound+tyrelife
deg_curves = df_filtered.groupby(['EventName', 'Compound', 'TyreLife'])['LapTimeDelta'].mean().reset_index()

# Build nested dict: {event: {compound: {tyrelife: delta}}}
deg_dict = {}
for event in deg_curves['EventName'].unique():
    event_df = deg_curves[deg_curves['EventName'] == event]
    deg_dict[event] = {}
    for compound in event_df['Compound'].unique():
        cmp_df = event_df[event_df['Compound'] == compound]
        deg_dict[event][compound] = {int(row['TyreLife']): round(float(row['LapTimeDelta']), 3) for idx, row in cmp_df.iterrows()}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(deg_dict, f, indent=4)

print(f"✅ Successfully analyzed and saved data-driven tyre degradation curves by track/compound to '{OUTPUT_FILE}'.")
