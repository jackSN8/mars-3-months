import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy import units as u

from poliastro.bodies import Sun, Earth, Mars
from poliastro.ephem import ephem
from poliastro.twobody import Orbit 
from poliastro.iod import lambert
from poliastro.util import norm

##############################################################################
# 1. Setup the date ranges
##############################################################################
departure_start = Time("2025-01-01", scale="tdb")
departure_end   = Time("2026-12-31", scale="tdb")  # e.g. 2-year window
n_departures    = 50
n_arrivals      = 50

min_flight_time = 100 * u.day
max_flight_time = 400 * u.day

# Generate departure dates
departure_dates = departure_start + (departure_end - departure_start) * np.linspace(0, 1, n_departures)

# For each departure, generate a range of arrival dates
arrival_dates = []
for dep_date in departure_dates:
    arr_start = dep_date + min_flight_time
    arr_end   = dep_date + max_flight_time
    arr_range = arr_start + (arr_end - arr_start) * np.linspace(0, 1, n_arrivals)
    arrival_dates.append(arr_range)
arrival_dates = np.array(arrival_dates)  # shape: (n_departures, n_arrivals)

##############################################################################
# 2. Create Ephem objects for Earth and Mars over the entire date range
##############################################################################
# We need to cover the entire time span that might be used: from the earliest departure to the latest arrival.
global_start = departure_start - 30*u.day
global_end   = departure_end + max_flight_time + 30*u.day

# Create an array of times spanning the entire window (for Ephem interpolation)
global_times = global_start + (global_end - global_start) * np.linspace(0, 1, 300)

mars_ephem = Ephem.from_body(Mars, global_times)
earth_ephem = Ephem.from_body(Earth, global_times)

##############################################################################
# 3. Solve Lambert’s problem over the grid
##############################################################################
delta_v_map = np.zeros((n_departures, n_arrivals))
tof_map     = np.zeros((n_departures, n_arrivals))

for i, dep_date in enumerate(departure_dates):
    for j, arr_date in enumerate(arrival_dates[i]):
        tof = (arr_date - dep_date).to(u.day).value
        tof_map[i, j] = tof
        
        if tof <= 0:
            delta_v_map[i, j] = np.nan
            continue
        
        # Get the position/velocity states from Ephem at these specific times
        r_dep, v_dep = mars_ephem.rv(dep_date)
        r_arr, v_arr = earth_ephem.rv(arr_date)
        
        # Solve the Lambert problem for Mars -> Earth
        try:
            (v_proposed_dep, v_proposed_arr), = lambert(Sun.k, r_dep, r_arr, (arr_date - dep_date))
            dv_depart = norm(v_proposed_dep - v_dep)
            dv_arrive = norm(v_proposed_arr - v_arr)
            total_dv  = dv_depart + dv_arrive
            delta_v_map[i, j] = total_dv.to(u.km/u.s).value
        except:
            delta_v_map[i, j] = np.nan

##############################################################################
# 4. (Optional) Plot a porkchop contour of total Delta-V
##############################################################################
fig, ax = plt.subplots(figsize=(10, 6))

dep_mjd = departure_dates.mjd
arr_mjd = np.array([[d.mjd for d in row] for row in arrival_dates])

valid_delta_v = delta_v_map[~np.isnan(delta_v_map)]
levels = np.linspace(valid_delta_v.min(), valid_delta_v.max(), 20)

cs = ax.contourf(dep_mjd, arr_mjd, delta_v_map.T, levels=levels, cmap='turbo')
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label(r'$\Delta V$ (km/s)')

ax.set_title("Mars-to-Earth Porkchop Plot (ΔV)")
ax.set_xlabel("Departure Date (MJD)")
ax.set_ylabel("Arrival Date (MJD)")

plt.tight_layout()
plt.show()
