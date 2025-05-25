# gui_routes_cards_with_coords.py

import os
import pickle
import math
import webbrowser

import customtkinter as ctk
import tkinter.messagebox as mb
import folium
import networkx as nx
import numpy as np

from datetime import datetime, date, timedelta
from tensorflow.keras.models import load_model

# ─── CONFIG ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SITES_PKL    = os.path.join(MODELS_DIR, 'sites_data.pkl')
SCALER_PATH  = os.path.join(MODELS_DIR, 'scaler_all.pkl')
MODEL_FILES = [
    'global_lstm_4sites.h5',
    'gru_model_all.h5',
    'rnn_model_all.h5'
]

SEQ_LEN      = 24
NUM_ROUTES   = 5
INTER_DELAY  = 30      # sec per intersection
FREE_SPEED   = 60.0    # km/h
# ──────────────────────────────────────────────────────────────────────────

# load site data & scaler once
try:
    sd        = pickle.load(open(SITES_PKL, 'rb'))
    sites_df  = sd['sites_df']
    flow_hist = sd['flow_hist']
    scaler    = pickle.load(open(SCALER_PATH, 'rb'))
except Exception as e:
    raise RuntimeError("Missing sites_data or scaler. Run create_sites_data.py first.") from e

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

def flow_to_time(flow, km):
    speed = FREE_SPEED if flow < 300 else max(5.0, FREE_SPEED*(1-(flow-300)/1000.0))
    return km/speed + INTER_DELAY/3600.0

def forecast_flows(model, depart_dt):
    flows = {}
    for sc, ts in flow_hist.items():
        cutoff = depart_dt - timedelta(hours=1)
        if cutoff < ts.index[0] or cutoff > ts.index[-1]:
            hist = ts.iloc[-SEQ_LEN:].values.reshape(-1,1)
        else:
            hist = ts.loc[:cutoff].iloc[-SEQ_LEN:].values.reshape(-1,1)
        if len(hist)<SEQ_LEN:
            hist = np.pad(hist, ((SEQ_LEN-len(hist),0),(0,0)), 'constant')
        s   = scaler.transform(hist)
        inp = s.reshape(1, SEQ_LEN,1).astype(np.float32)
        pred= model.predict(inp, verbose=0)
        flows[sc] = scaler.inverse_transform(pred)[0,0]
    return flows

def suggest_routes(origin, dest, hhmm, model_file):
    model = load_model(os.path.join(MODELS_DIR, model_file), compile=False)
    depart_dt = datetime.combine(date.today(), datetime.strptime(hhmm, "%H:%M").time())
    flows = forecast_flows(model, depart_dt)

    G = nx.DiGraph()
    for u in sites_df.index:
        for v in sites_df.index:
            if u==v: continue
            lat1, lon1 = sites_df.at[u,'lat'], sites_df.at[u,'lon']
            lat2, lon2 = sites_df.at[v,'lat'], sites_df.at[v,'lon']
            d_km = haversine(lat1,lon1,lat2,lon2)
            t_h  = flow_to_time(flows[v], d_km)
            G.add_edge(u, v, time_h=t_h, distance_km=d_km)

    if origin==dest:
        return [([origin], 0.0, 0.0)]

    paths = nx.shortest_simple_paths(G, origin, dest, weight='time_h')
    seen, out = set(), []
    for path in paths:
        tp = tuple(path)
        if tp in seen: continue
        seen.add(tp)
        total_h  = sum(G[u][v]['time_h'] for u,v in zip(path,path[1:]))
        total_km = sum(G[u][v]['distance_km'] for u,v in zip(path,path[1:]))
        out.append((list(path), total_h*60, total_km))
        if len(out)>=NUM_ROUTES: break
    return out

# === MAP FUNCTION ===
def show_on_map(routes):
    # center on origin of first route
    origin = routes[0][0][0]
    lat0, lon0 = sites_df.at[origin,'lat'], sites_df.at[origin,'lon']
    fmap = folium.Map(location=[lat0, lon0], zoom_start=13)

    colors = ['green'] + ['gray']*(len(routes)-1)
    for idx, (route, _, _) in enumerate(routes):
        coords = [(sites_df.at[s,'lat'], sites_df.at[s,'lon']) for s in route]
        folium.PolyLine(coords,
                        color=colors[idx],
                        weight=5,
                        opacity=0.8,
                        tooltip=f"Route {idx+1}"
                       ).add_to(fmap)
        # Mark start & end
        folium.CircleMarker(coords[0], radius=6, color='blue',
                            popup=f"Start: {route[0]}").add_to(fmap)
        folium.CircleMarker(coords[-1], radius=6, color='red',
                            popup=f"End: {route[-1]}").add_to(fmap)

    outpath = os.path.abspath("routes_map.html")
    fmap.save(outpath)
    webbrowser.open(f"file://{outpath}")

# === GUI ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("TBRGS Route Suggestions")
root.geometry("600x750")

ctk.CTkLabel(root, text="Traffic‐Based Route Guidance",
             font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

# controls
frame_ctrl = ctk.CTkFrame(root)
frame_ctrl.pack(fill="x", padx=20, pady=(0,10))

origin_cb = ctk.CTkComboBox(frame_ctrl,
    values=[str(int(s)) for s in sites_df.index])
origin_cb.set("Origin")
origin_cb.pack(side="left", expand=True, padx=5)

dest_cb = ctk.CTkComboBox(frame_ctrl,
    values=[str(int(s)) for s in sites_df.index])
dest_cb.set("Destination")
dest_cb.pack(side="left", expand=True, padx=5)

time_ent = ctk.CTkEntry(frame_ctrl, placeholder_text="HH:MM")
time_ent.pack(side="left", expand=True, padx=5)

model_cb = ctk.CTkComboBox(frame_ctrl, values=MODEL_FILES)
model_cb.set("Model")
model_cb.pack(side="left", expand=True, padx=5)

btn = ctk.CTkButton(frame_ctrl, text="Compute Routes")
btn.pack(side="left", expand=True, padx=5)

# results area
res_frame = ctk.CTkScrollableFrame(root, width=560, height=600)
res_frame.pack(padx=20, pady=10, fill="both", expand=True)

def on_compute():
    for child in res_frame.winfo_children():
        child.destroy()
    try:
        o  = int(origin_cb.get())
        d  = int(dest_cb.get())
        t  = time_ent.get().strip()
        mf = model_cb.get().strip()
        if not all([o,d,t,mf]):
            raise ValueError("Please select all fields.")
        routes = suggest_routes(o, d, t, mf)

        for idx,(route, tmin, dist) in enumerate(routes, start=1):
            fg = "#27ae60" if idx == 1 else "#2c3e50"

            card = ctk.CTkFrame(res_frame, fg_color=fg, corner_radius=10)
            card.pack(fill="x", pady=5, padx=5)

            parts = []
            for s in route:
                lat, lon = sites_df.at[s,'lat'], sites_df.at[s,'lon']
                parts.append(f"{s} ({lat:.4f}, {lon:.4f})")
            route_str = " → ".join(parts)

            text = (
                f"Route {idx}: {route_str}\n"
                f"  • Time:        {tmin:.1f} min\n"
                f"  • Distance:    {dist:.2f} km"
            )
            ctk.CTkLabel(
                card, text=text, justify="left",
                font=ctk.CTkFont(size=14)
            ).pack(padx=10, pady=10)

        # add the map button
        map_btn = ctk.CTkButton(
            res_frame,
            text="Show on Map 🗺️",
            fg_color="#3498db",
            hover_color="#2980b9",
            command=lambda: show_on_map(routes)
        )
        map_btn.pack(pady=10)

    except Exception as e:
        mb.showerror("Error", str(e))

btn.configure(command=on_compute)

root.mainloop()
