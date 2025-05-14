import numpy as np

def flow_to_speed(flow):
    if flow <= 351:
        return 60.0  # Speed limit
    else:
        a = -1.4648375
        b = 93.75
        c = -flow
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None  
        # Two roots, pick the smaller one (congested)
        speed1 = (-b + np.sqrt(discriminant)) / (2*a)
        speed2 = (-b - np.sqrt(discriminant)) / (2*a)
        return min(speed1, speed2)

def calculate_travel_time(flow, distance_km, intersections=1):
    speed = flow_to_speed(flow)
    if speed is None:
        return None
    travel_time = distance_km / speed  # in hours
    travel_time += (30 * intersections) / 3600  # convert 30s delay per intersection to hours
    return travel_time * 60  # return in minutes