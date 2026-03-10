import zipfile
import pandas as pd
import os
raw_zip = "data/ets_gtfs.zip"
processed_csv = "data/processed/training_data.csv"

os.makedirs("data/processed", exist_ok=True)
with zipfile.ZipFile(raw_zip, 'r')as z:
    # Force IDs to strings to avoid the DtypeWarning and data corruption
    stops = pd.read_csv(z.open('stops.txt'), dtype={'stop_id': str})
    routes = pd.read_csv(z.open('routes.txt'), dtype={'route_id': str})
    trips = pd.read_csv(z.open('trips.txt'), dtype={'trip_id': str, 'route_id': str, 'service_id': str})
    stop_times = pd.read_csv(z.open('stop_times.txt'), dtype={'trip_id': str, 'stop_id': str}, low_memory=False)
    calendar = pd.read_csv(z.open('calendar_dates.txt'))
df = stop_times.merge(trips,on='trip_id', how='left').merge(routes, on='route_id', how='left')
df = df.merge(stops[['stop_id','stop_lat','stop_lon']], on='stop_id', how='left')
df.to_csv(processed_csv,index=False)
print(f"Processed data saved to {processed_csv}")