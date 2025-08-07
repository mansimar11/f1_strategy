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
