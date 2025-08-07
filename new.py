# verify_data.py
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL

print("Connecting to the database to verify data...")

try:
    engine = create_engine(DATABASE_URL)

    # Query 1: Get the total number of rows
    total_rows_df = pd.read_sql("SELECT COUNT(*) FROM laps;", engine)
    total_rows = total_rows_df.iloc[0, 0]
    print(f"\n✅ Total rows in 'laps' table: {total_rows}")

    # Query 2: Get the count of laps per year
    print("\n✅ Lap count per year:")
    laps_per_year_df = pd.read_sql(
        'SELECT "Year", COUNT(*) as "LapCount" FROM laps GROUP BY "Year" ORDER BY "Year";',
        engine
    )
    print(laps_per_year_df.to_string(index=False))

    # Query 3: Show a small sample of the data
    print("\n✅ Data sample:")
    sample_df = pd.read_sql("SELECT * FROM laps LIMIT 5;", engine)
    print(sample_df.to_string())

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please ensure your config.py is using the 'External Database URL' from Render.")