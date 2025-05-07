import pandas as pd
import os

# File paths
site_file = os.path.join("data", "SCATSSiteListingSpreadsheet_VicRoads.xls")
coord_file = os.path.join("data", "Traffic_Count_Locations_with_LONG_LAT.csv")

# Load SCATS site listing (SiteID + intersection name)
site_df = pd.read_excel(site_file)

# Load location coordinates (SiteID + lat/lon)
coord_df = pd.read_csv(coord_file)

# Clean up column names
site_df.columns = [str(col).strip() for col in site_df.columns]
coord_df.columns = [str(col).strip() for col in coord_df.columns]

# Merge the two files using the site ID (check if columns match exactly)
merged_df = pd.merge(site_df, coord_df, how='inner', on='Site ID')

# Filter and rename relevant columns
final_df = merged_df[['Site ID', 'Intersection Name', 'Latitude', 'Longitude']].copy()
final_df.columns = ['SiteID', 'Name', 'Lat', 'Lon']

# Save to CSV
output_file = os.path.join("data", "scats_site_info.csv")
final_df.to_csv(output_file, index=False)

print(f"\n✅ SCATS site info saved to: {output_file}")
print(final_df.head())
