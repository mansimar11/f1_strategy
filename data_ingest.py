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