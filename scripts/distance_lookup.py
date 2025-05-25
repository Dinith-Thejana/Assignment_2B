import pandas as pd
import math
import json

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Load the cleaned CSV
df = pd.read_csv('../data/traffic_cleaned.csv')

# Drop rows with missing coordinates
df = df.dropna(subset=['NB_latitude', 'NB_longitude'])

# Build SCATS coordinate dictionary
scats_coords = df.groupby('SCATS Site')['NB latitude', 'NB longitude'].first().to_dict('index')

# Build distance lookup dictionary
distance_lookup = {}
scats_ids = list(scats_coords.keys())

for origin in scats_ids:
    for dest in scats_ids:
        if origin != dest:
            lat1, lon1 = scats_coords[origin]['NB latitude'], scats_coords[origin]['NB longitude']
            lat2, lon2 = scats_coords[dest]['NB latitude'], scats_coords[dest]['NB longitude']
            distance = haversine(lat1, lon1, lat2, lon2)
            distance_lookup[(origin, dest)] = round(distance, 3)

# Save to JSON
with open('data/distance_lookup.json', 'w') as f:
    json.dump({f"{k[0]}-{k[1]}": v for k, v in distance_lookup.items()}, f, indent=2)

print("✅ distance_lookup.json generated!")
