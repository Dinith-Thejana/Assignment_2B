import pandas as pd
import os

# Load the Excel file and "Data" sheet
df = pd.read_excel(r"..\data\Scats Data October 2006.xls", sheet_name="Data", header=1)
df.head

# Clean column names
df.columns = [str(col).strip() for col in df.columns]

# Print column names (for confirmation/debugging)
print(" Columns found:")
print(df.columns)

# Identify ID and time columns
fixed_cols = ['SCATS Number', 'Location']  # Adjust if needed
time_cols = [col for col in df.columns if ':' in col or col.strip().endswith("AM") or col.strip().endswith("PM")]

# Melt to long format (flatten the time columns)
df_long = df.melt(id_vars=fixed_cols, value_vars=time_cols,
                  var_name='Time', value_name='VehicleCount')

# Add a dummy date
df_long['Date'] = '2006-10-01'

# Combine Date + Time into datetime
df_long['Datetime'] = pd.to_datetime(df_long['Date'] + ' ' + df_long['Time'], errors='coerce')

# Drop invalid datetimes and clean up
df_long.drop(columns=['Date', 'Time'], inplace=True)
df_long.dropna(subset=['Datetime', 'VehicleCount'], inplace=True)

# Rename and set index
df_long.rename(columns={'SCATS Number': 'SiteID'}, inplace=True)
df_long.set_index('Datetime', inplace=True)

# Resample into hourly totals
df_hourly = df_long.groupby(['SiteID']).resample('1H').sum().reset_index()

# Save result
output_path = os.path.join("data", "cleaned_traffic_data.csv")
df_hourly.to_csv(output_path, index=False)

print(f"\ Cleaned hourly traffic data saved to: {output_path}")
