<<<<<<< HEAD
import fastf1
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import pytz
import os
import time

# -------- CONFIG --------
DB_CONNECTION_STRING = 'postgresql://neondb_owner:npg_6lYhEpGDCmW1@ep-orange-cloud-a1q1wej3-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require'
CACHE_DIR = '.fastf1_cache'
TABLE_NAME = 'f1_live_data'
START_YEAR = 2022
CURRENT_YEAR = datetime.now().year
# ------------------------

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

engine = create_engine(DB_CONNECTION_STRING)

all_laps = []

for year in range(START_YEAR, CURRENT_YEAR + 1):
    print(f"\n📅 Processing year: {year}")
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['Session5Date'] < datetime.now(pytz.utc)]
    except Exception as e:
        print(f"❌ Could not get schedule for {year}: {e}")
        continue

    for _, race in schedule.iterrows():
        gp_name = race['EventName']
        print(f"🏁 Loading race: {year} {gp_name}")
        try:
            session = fastf1.get_session(year, gp_name, 'R')
            session.load()
            laps = session.laps
            laps['SessionType'] = 'R'

            if laps is None or laps.empty:
                print(f"⚠️ No lap data for {gp_name}. Continuing.")
                continue

            # Enrich data
            laps['Year'] = year
            laps['EventName'] = gp_name
            laps['driver_position'] = laps.get('Position', None)
            laps['team'] = laps.get('Team', None)
            laps['driver'] = laps.get('Driver', None)
            laps['compound'] = laps.get('Compound', None)
            laps['stint'] = laps.get('Stint', None)
            laps['track_status'] = laps.get('TrackStatus', None)
            laps['pit_in'] = laps.get('PitIn', None)
            laps['pit_out'] = laps.get('PitOut', None)
            laps['tyre_age'] = laps.get('TyreLife', None)
            laps['lap_time_sec'] = laps['LapTime'].dt.total_seconds() if laps['LapTime'].notna().any() else None
            laps['is_personal_best'] = laps.get('IsPersonalBest', None)

            # Driver rank by fastest lap
            try:
                fastest = laps.groupby('Driver')['LapTime'].min().sort_values().reset_index()
                fastest['driver_rank'] = fastest.index + 1
                laps = pd.merge(laps, fastest[['Driver', 'driver_rank']], on='Driver', how='left')
            except Exception:
                laps['driver_rank'] = None

            # Team pace average
            try:
                team_pace = laps.groupby('Team')['lap_time_sec'].mean().reset_index()
                team_pace.rename(columns={'lap_time_sec': 'avg_team_lap_time'}, inplace=True)
                laps = pd.merge(laps, team_pace, on='Team', how='left')
            except Exception:
                laps['avg_team_lap_time'] = None

            # Clean columns
            final_cols = [
                'Year', 'EventName', 'Driver', 'Team', 'LapNumber', 'lap_time_sec', 'Stint',
                'Compound', 'TyreLife', 'Position', 'driver_position', 'track_status', 'PitIn',
                'PitOut', 'is_personal_best', 'driver_rank', 'avg_team_lap_time'
            ]
            for col in final_cols:
                if col not in laps.columns:
                    laps[col] = None

            laps = laps[final_cols]
            laps.dropna(subset=['lap_time_sec'], inplace=True)

            all_laps.append(laps)
            print(f"✅ Added {len(laps)} laps from {gp_name}")

        except Exception as e:
            print(f"⚠️ Error during {gp_name}: {e} — continuing without skipping.")
        time.sleep(1)

# Combine all laps and save
if not all_laps:
    print("❌ No valid race data collected. Exiting.")
    exit()

combined_df = pd.concat(all_laps, ignore_index=True)
print(f"\n💾 Uploading {len(combined_df)} rows to NeonDB...")

combined_df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
print("✅ Upload complete! Data is ready in Power BI and for ML models.")
=======
# ingest_data.py
import fastf1
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL
import time
from datetime import datetime
from pathlib import Path

# --- 1. SETUP ---
# Define a start year and get the current year automatically.
START_YEAR = 2022
CURRENT_YEAR = datetime.now().year
YEARS_TO_PROCESS = list(range(START_YEAR, CURRENT_YEAR + 1))

# Connect to your PostgreSQL database
engine = create_engine(DATABASE_URL)
print("✅ Connection to database successful.")

# Create a path to a cache directory in your home folder that works on any OS
cache_dir = Path.home() / 'fastf1_cache'
# Create the directory if it doesn't exist
cache_dir.mkdir(parents=True, exist_ok=True)
# Tell fastf1 where to store its cache
fastf1.Cache.enable_cache(cache_dir)
print(f"✅ FastF1 cache enabled at: {cache_dir}")


# --- 2. MAIN SCRIPT ---
all_laps_data = [] # Create an empty list to hold data from all races

print(f"\nFetching race data for the years: {YEARS_TO_PROCESS}...")

# Loop through each year
for year in YEARS_TO_PROCESS:
    # --- THIS IS THE ADDED FIX ---
    # Wrap schedule loading in a try-except block to handle unavailable years
    try:
        schedule = fastf1.get_event_schedule(year)
    except ValueError:
        print(f"Could not fetch schedule for {year}. Skipping.")
        continue # Move to the next year in the loop
    # ---------------------------
    
    # We only want to process events that have already happened
    races_to_process = schedule[schedule['EventDate'] < pd.to_datetime('today')]
    
    # Check if there are any past events for the given year
    if len(races_to_process) == 0:
        print(f"No past events found for {year}. Skipping.")
        continue
        
    print(f"Found {len(races_to_process)} past events for {year}.")

    # Loop through each race in the schedule
    for _, event in races_to_process.iterrows():
        try:
            print(f"    [DEBUG] Starting session load for {year} {event['EventName']}")
            session = fastf1.get_session(year, event['EventName'], 'R') # 'R' stands for Race
            print(f"    [DEBUG] Calling session.load for {year} {event['EventName']}")
            session.load(telemetry=False, weather=False)
            print(f"    [DEBUG] session.load finished for {year} {event['EventName']}")

            # Get the laps data into a pandas DataFrame
            laps_df = session.laps

            # Add columns to identify the year and event
            laps_df['Year'] = year
            laps_df['EventName'] = event['EventName']

            # Append the data for this race to our main list
            all_laps_data.append(laps_df)

            print(f"  -> Successfully processed {year} {event['EventName']}")

        except Exception as e:
            # Some sessions might fail to load (e.g., data not available), so we just skip them
            print(f"  -> Could not process {year} {event['EventName']}. Reason: {e}")
        
        # A small delay to be respectful to the API server
        time.sleep(1) 

# --- 3. SAVE TO DATABASE ---
if all_laps_data:
    # Combine the data from all races into a single large DataFrame
    final_df = pd.concat(all_laps_data, ignore_index=True)
    
    # Select and rename the columns we want to save
    final_df = final_df[['Year', 'EventName', 'Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound', 'TyreLife']]
    final_df = final_df.rename(columns={'Driver': 'DriverCode'})

    # Convert LapTime (a timedelta object) to total seconds for easier analysis
    final_df['LapTime'] = final_df['LapTime'].dt.total_seconds()
    
    # Handle potential missing values before saving
    final_df.dropna(subset=['LapTime'], inplace=True)
    
    print("\nSaving all data to the database... This may take a few minutes.")
    # Save the final DataFrame to the 'laps' table in your database.
    # if_exists='replace' will create a fresh table every time you run the script.
    final_df.to_sql('laps', engine, if_exists='replace', index=False)
    
    print("✅ All data has been successfully saved to the database!")
else:
    print("\nNo new data was fetched. The database was not changed.")
>>>>>>> 6654cc450a78f302f5fe365ff121aa506498f2a0
