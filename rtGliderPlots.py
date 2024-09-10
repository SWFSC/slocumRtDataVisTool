# important glider variable to plot:
    # depth
        # dive/climb speed --> linear regression of both sides
        # alert if slopes not good
        # aleart target depth exceded 
            # tell how much overshot
    # Pitch/roll
    # Location
        # distance from ideal track and target
    # temp
    # sal
    # O2
    # 

import numpy as np
import pandas as pd
import xarray as xr 
import dbdreader
import matplotlib.pyplot as plt
import os
import scipy as sp
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from numpy.linalg import lstsq
import gsw

# read in data from netCDF file into object named data
data_path = "/Users/cflaim/Documents/data/test/amlr01-20181216/amlr01-20181216-union.nc"
data = xr.open_dataset(data_path)

# netCDF variable names for reference
# vars = ["time", "heading", "pitch", "roll", "m_depth", "waypoint_latitude", "waypoint_longitude", 
#         "conductivity", "temperature", "pressure", "chlorophyll", "cdom", "backscatter_700", 
#         "oxygen_concentration", "distance_over_ground", "salinity", "potential_density",
#         "profile_index", "profile_direction"]

# Create variables for time, depth, and profile indice
reg_time = np.float64(data['time'].values)
time = data['time'].values
depth = data["m_depth"].values
prof_inds = data['profile_index'].values
pitch = data['pitch'].values
roll = data['roll'].values

# Limit data to first 50k entries for ease of developement
reg_time = reg_time[0:10000]
time = time[0:10000]
depth = depth[0:10000]
prof_inds = prof_inds[0:10000]
pitch = pitch[0:10000]
roll = roll[0:10000]

gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 0.25, 0.25])
fig = plt.figure(constrained_layout = True, figsize=(11, 8.5))
gs = fig.add_gridspec(6, 7)
ax1 = fig.add_subplot(gs[0:4, :5])
ax2 = fig.add_subplot(gs[4, :5], sharex=ax1)
ax3 = fig.add_subplot(gs[5, :5], sharex=ax2)
ax4 = fig.add_subplot(gs[:, 5:], fc='wheat', alpha=0.5)
# ax5 = fig.add_subplot(gs[4:, 5:], fc='wheat', alpha=0.5)

ax4.xaxis.set_tick_params(labelbottom=False)
ax4.yaxis.set_tick_params(labelleft=False)
ax4.set_xticks([])
ax4.set_yticks([])

# ax5.xaxis.set_tick_params(labelbottom=False)
# ax5.yaxis.set_tick_params(labelleft=False)
# ax5.set_xticks([])
# ax5.set_yticks([])
# fig, [ax1, ax2, ax3]= plt.subplots(3,1, figsize=(11,8.5), sharex=True, gridspec_kw=gs_kw)
ax1.set_title(f"Profiles {np.unique(prof_inds)[0]} - {np.unique(prof_inds)[-2]}")
ax1.invert_yaxis()
y2=ax1.twinx()
ax1.set_ylabel("Depth [m]")
ax2.set_ylabel("Pitch [deg]")
ax3.set_ylabel("Roll [deg]")
ax3.set_xlabel("Time")

w_s = [] # empty list to store vertical velocities from each profile
v_gs = [] # empty list to store velocity over ground from each profile
v_ps = [] # empty list to store velocity along path from each profile
pitch_means = []

# Create dictionary to hold start and end index for each profile index
dive_segs = {} 
# follows structure of {'prof_name' : {'start_ind': #, 'end_ind': #, 
#                                       'w': {'climb': #, 'dive': #}
#                                       'v_p': {'climb': #, 'dive':#}
#                                       'v_g': {'climb': #, 'dive':#}
#                                       'pitch': {'climb': #, 'dive':#}
#                                       'roll': {'climb': #, 'dive': #}
#                                       '} }
for val in np.unique(prof_inds)[0:len(np.unique(prof_inds))-1]: # fill above 'dive_segs' dict with values
    if val != np.float64('nan'): 
        start = np.where(prof_inds==val)[0][0]
        end = np.where(prof_inds==val)[0][-1]

        d_sub = depth[start:end]
        d_filt = d_sub > 1
        d_sub = d_sub[d_filt]
        d_sub = d_sub.reshape(-1,1)

        t_sub = reg_time[start:start+len(d_sub)].reshape(-1,1)

        r_sub_raw = roll[start:end]

        model = LinearRegression()
        model.fit(t_sub, d_sub)
        d_pred = model.predict(t_sub)
        w = -1*model.coef_[0][0]*np.power(10,9) # convert to m/s from m/ns

        p_sub_raw = pitch[start:end]*180/np.pi
        if -1*model.coef_[0][0] > 0:
            p_pos = p_sub_raw > 0
            p_sub_pos = p_sub_raw[p_pos]
            p_sub_mean = np.nanmean(p_sub_pos)
            pitch_means.append(p_sub_mean)
        else:
            p_neg = p_sub_raw < 0
            p_sub_neg = p_sub_raw[p_neg]
            p_sub_mean = np.nanmean(p_sub_neg)
            pitch_means.append(p_sub_mean)

        vel_along_path = w/np.cos(p_sub_mean) # ISSUE
        vel_over_ground = w*np.tan(p_sub_mean) # ISSUE

        seg_name = f'prof{val}'
        dive_segs[seg_name] = {'start_ind':start, 'end_ind':end, 'w': w, 'v_g':vel_over_ground, 
                               'v_p':vel_along_path}
        
        ax1.scatter(time[start:end], depth[start:end], s=1.5)
        ax1.plot(time[start:start+len(d_sub)], d_pred, label=f"{seg_name[0:-2]} w ≈ {w:0.3f} m/s")

        ax2.scatter(time[start:end], p_sub_raw, s=2, label='')
        ax3.scatter(time[start:end], r_sub_raw, s=1, label='')

        w_s.append(w)
        v_gs.append(vel_over_ground)
        v_ps.append(vel_along_path)

w_s = np.array(w_s) 
v_gs = np.array(v_gs)
v_ps = np.array(v_ps)
pitch_means = np.array(pitch_means)

w_pos = w_s > 0
w_neg = w_s < 0
w_climb_mean = np.nanmean(w_s[w_pos])
w_dive_mean = np.nanmean(w_s[w_neg])

v_p_pos = v_gs > 0
v_p_neg = v_gs < 0
vp_climb_mean = np.nanmean(v_gs[v_p_pos])
vp_dive_mean = np.nanmean(v_gs[v_p_neg])

v_g_pos = v_gs > 0
v_g_neg = v_gs < 0
vg_climb_mean = np.nanmean(v_gs[v_g_pos])
vg_dive_mean = np.nanmean(v_gs[v_g_neg])

p_pos = pitch_means > 0
p_neg = pitch_means < 0
p_climb_mean = np.nanmean(pitch_means[p_pos])
p_dive_mean = np.nanmean(pitch_means[p_neg])

# check all values if w/in range for variable
# Have a defualt string to write to screen if nothing wrong
# If variable or variables out of range, move to top, make bold, italic, and red

# write function/code to loop thru dicts of mean values and acceptable ranges
    # produce formatted string based on acceptable or not
    # return long string with all of the needed formatting

    # OOOOORRRRRR
    # Use as much of default string as possible
    # only formated needed parts and truncate default string

pitch_string = f"Avg climb angle: {p_climb_mean:0.3f}°\nAvg dive angle: {p_dive_mean:0.3f}°\n\n"
info_string = f"\
Avg climb roll: {0}°\nAvg dive roll: {0}°\n\n\
Avg climb w: {w_climb_mean:0.3f} m/s\nAvg dive w: {w_dive_mean:0.3f} m/s\n\n\
Avg climb v_p: {vp_climb_mean:0.3f} m/s\nAvg dive v_p: {vp_dive_mean:0.3f} m/s\n\n\
Avg climb v_g: {vg_climb_mean:0.3f} m/s\nAvg dive v_g: {vg_dive_mean:0.3f} m/s\n\n\
Estimated distance traveled: {2}\n\
GPS distance traveled: {0}\n\n"

ax4.text(0.05, 0.98, pitch_string, verticalalignment='top', c = 'r')
ax4.text(0.05, 0.91, info_string, verticalalignment='top')

apogees, _ = sp.signal.find_peaks(depth, height=(-10, None))

ax1.legend(loc="lower left", ncol=3)
plt.xticks(rotation=15)
plt.tight_layout()
# plt.savefig('test.png')
plt.show()


























def getWorkingDir():
    pass

def inputDiveParams():
    pass

def loadRaw(): # needs to be pointed to working directory
    pass

def convRaw(): # look at Sam's code for example
    pass

def smoothData(): # rolling average?
    pass

def makePlots(): # needs to be pointed to a save directory
    pass

def run():
    loadRaw()
    convRaw()
    smoothData()
    makePlots()


if __name__ == "plotGliderPlots":
    run()
