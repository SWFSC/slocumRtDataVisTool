# Data modules
import numpy as np
import pandas as pd
import xarray as xr 
import dbdreader
import matplotlib.pyplot as plt
import os
import scipy as sp
from scipy import odr
from sklearn.linear_model import LinearRegression
import gsw
import matplotlib.dates as mdates
import cmocean.cm as cmo
from mpl_toolkits.basemap import Basemap
import pickle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from datetime import datetime
import zipfile
from google.cloud import secretmanager
import google_crc32c
import pyglider.utils as pgu
import esdglider.gcp as gcp

# Email modules
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Function to zip everything in a folder
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

def returnNan(t, v):
    return lambda _t: np.NaN

#need to mount buckets
#need to access gcp secret for email pw
class gliderData:
    def __init__(self):
        self.data_dir = ""
        self.cache_dir = ""
        self.glider = ""
        self.date = ""
        self.max_amphrs = 800
        self.n_yos = -1
        self.n_yos_tot = -1
        self.dive_start = -1
        self.dive_end = -1
        self.dep_start = -1
        self.dep_end = -1

        self.tm = 0
        self.depth = 0
        self.roll = 0
        self.pitch = 0
        self.oil_vol = 0
        self.pressure = 0
        self.temp = 0
        self.cond = 0
        self.lat = 0
        self.lon = 0
        self.sea_pressure = 0
        self.vacuum = -1
        self.amphr = 0
        self.o2 = 0
        self.cdom = 0
        self.chlor = 0
        self.backscatter = 0
        self.this_call = "call_0"

  
        self.problem_dives = [] # empty list to store number values of problematic dives
        
        self.profiles_to_make = {"profile0":["cdom"],
                                 "profile1":["absolute_salinity", "temp", "sigma"],
                                 "profile2":["oxygen"],
                                 "profile3":["chlorophyl"]} # Dictionary to be looped thru to make depth profiles of scienec vars
      
        self.sci_vars = ["cdom", "chlorophyl", "oxygen", "backscatter"] # List of science variables
        self.sci_colors = {"cdom":cmo.solar, "chlorophyl": cmo.algae, "oxygen":cmo.tempo, "backscatter":cmo.haline} # dictionary of colormaps to be used for listed science variables

        # Dictionary used to store strings of processed data stats and 
        # parameters to display the string (e.g., color, condasize)
        self.data_strings = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : ("w = ### m/s", 14, 'r')}}
        
        # Empty dictionary to store data for dives/climbs
        self._data = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : [list of values, e.g., roll]}}
        # Dictionary of units for plotting purposes
        self.units = {"w":'m/s', "pitch":'deg', "roll":'deg', "dt":'hrs', "amphr":'Ahr', "vac":'inHg', "cdom":"nanometers", "chlorophyl":"µmol•$m^{-2}$", "oxygen":"mL•$L^{-1}$", "backscatter":"$m^{-1}$"} # 
        # Dictionary to store acceptable ranges fpr flight parameters
        self.acceptable_ranges = {"dive":{"w":[-0.08, -0.20], "pitch":[-20, -29], "roll":[-5, 5], "dt":[0.25, 3], "amphr":[0.05, 1.0], "vac":[6, 10]},
                                  "climb":{"w":[0.08, 0.20], "pitch":[20, 29], "roll":[-5, 5], "dt":[0.25, 3], "amphr":[0.05, 1.0], "vac":[6, 10]}}
      
    def getWorkingDirs(self):
        self.data_dir = "data/new_data/" # set directoy path for data
        self.cache_dir = "/Users/cflaim/Documents/GitHub/standard-glider-files/Cache/" # set directoy path for cache

        # Grabs the glider's name from the data files
        glider = os.listdir(self.data_dir)
        for g in glider:
            if ".tbd" in g:
                self.glider = g.split("-")[0]
                break

    def readRaw(self):
        # interp_fact_keys = ["m_depth", 'sci_flbbcd_bb_units','sci_flbbcd_chlor_units', 'sci_flbbcd_cdom_units','sci_oxy4_oxygen','m_coulomb_amphr_total', 'm_vacuum','m_roll', 'm_pitch', 'm_de_oil_vol', 'sci_water_pressure', 'sci_water_temp', 'sci_water_cond', 'm_lat', 'm_lon']
        # interp_fact = {}
        # for key in interp_fact_keys:
        #     interp_fact[key] = returnNan

        # Creates dbd reader object
        dbd = dbdreader.MultiDBD(pattern=self.data_dir+'*.[st]bd', cacheDir=self.cache_dir, complemented_files_only=True)

        # Set class variables to data arrays from dbdreader. 
        # NOTE: order of these is arbitrary, but depth must be second!
        self.tm, self.depth, self.backscatter, self.chlor, self.cdom, self.o2, self.amphr, self.vacuum, self.roll, self.pitch, self.oil_vol, self.pressure, self.temp, self.cond, self.lat, self.lon = dbd.get_sync("m_depth", 'sci_flbbcd_bb_units','sci_flbbcd_chlor_units', 'sci_flbbcd_cdom_units','sci_oxy4_oxygen','m_coulomb_amphr_total', 'm_vacuum','m_roll', 'm_pitch', 'm_de_oil_vol', 'sci_water_pressure', 'sci_water_temp', 'sci_water_cond', 'm_lat', 'm_lon', interpolating_function_factory=None)
        
        self.roll, self.pitch = self.roll*180/np.pi, self.pitch*180/np.pi # convert roll and pitch from rad to deg
        self.sea_pressure = self.pressure * 10 - 10.1325 # convert pressure to sea pressure in dbar for GSW
        self.cond = self.cond * 10 # convert cond from s/m to mS/cm for GSW

        # Uncomment to print out available sci and eng variables
        # print("we the following science parameters:")
        # for i,p in enumerate(dbd.parameterNames['sci']):
        #     print("%2d: %s"%(i,p))
        # print("\n and engineering paramters:")
        # for i,p in enumerate(dbd.parameterNames['eng']):
        #     print("%2d: %s"%(i,p))

        dbd.close()

    def makeDf(self): 
        # creates temporary time list after converting time from ns to datetime strings
        _dt = np.array([dbdreader.dbdreader.epochToDateTimeStr(t, timeformat='%H:%M:%S') for t in self.tm])
        time = np.array([t[0]+ ' ' +t[1] for t in _dt]) # combines separate data and time strings into a singular string

        vars = [time, self.depth, self.backscatter, self.chlor, self.cdom, self.o2, self.amphr, self.vacuum, self.roll, self.pitch, self.oil_vol, self.sea_pressure, self.pressure, self.temp, 
                self.cond, self.lat, self.lon] # List of variable values for pandas DF
        columns = ['time', 'depth_m', 'backscatter', 'chlorophyl', 'cdom', 'oxygen', 'amphr', 'vacuum', 'roll_deg', 'pitch_deg', 'oil_vol','sea_pressure', 'pressure', 
                'temp', 'cond', 'lat', 'lon'] # List of column names for pandas df
        
        df_dict = {} # empty dict to be filled 
        # fill above dictionary
        for i, var in enumerate(columns):
            df_dict[var] = vars[i]

        self.df = pd.DataFrame(df_dict) # make pandas df from dict
        self.df['time'] = pd.to_datetime(self.df['time']) # convert the time column to DateTime type from datestrings
        self.date = np.max(self.df.time)
        self.date = datetime.strftime(self.date, "%Y-%b-%d") # convert time strings to datetime

        # NOTE: should do somehting here to check for gaps in the data to avoid 
        self.df = self.df.query("cond > 0")
        self.df = self.df.query("temp > -1")
        self.df = self.df.query("sea_pressure > -1")
        self.df.reindex()

    def getProfiles(self):
        # Code taken and modified from pyglider to work with pandas
        min_dp=10.0
        inversion=3.
        filt_length=7
        min_nsamples=14

        profile = self.df.pressure.values * np.nan
        direction = self.df.pressure.values * np.nan
        pronum = 1
        lastpronum = 0

        good = np.where(~np.isnan(self.df.pressure.values))[0]
        p = np.convolve(self.df.pressure.values[good],
                        np.ones(filt_length) / filt_length, 'same')
        dpall = np.diff(p)
        inflect = np.where(dpall[:-1] * dpall[1:] < 0)[0]
        for n, i in enumerate(inflect[:-1]):
            try:
                nprofile = inflect[n+1] - inflect[n]
                inds = np.arange(good[inflect[n]], good[inflect[n+1]]+1) + 1
                dp = np.diff(self.df.pressure[inds[[-1, 0]]])

                if ((nprofile >= min_nsamples) and (np.abs(dp) > 10)):
                    direction[inds] = np.sign(dp)
                    profile[inds] = pronum
                    lastpronum = pronum
                    pronum += 1
                else:
                    profile[good[inflect[n]]:good[inflect[n+1]]] = lastpronum + 0.5

            except KeyError:
                continue

        self.df['profile_index'] = profile
        self.df['profile_direction'] = direction

        if "new" in self.data_dir:
            self.n_yos = np.max(self.df.profile_index) - np.min(self.df.profile_index)
            self.dive_start = datetime.strftime(np.min(self.df.time), "%Y-%b-%d %H:%M")
            self.dive_end = datetime.strftime(np.max(self.df.time), "%Y-%b-%d %H:%M")
        else:
            self.n_yos_tot = np.max(self.df.profile_index)
            self.dep_start = datetime.strftime(np.min(self.df.time), "%Y-%b-%d %H:%M")
            self.dep_end = datetime.strftime(np.max(self.df.time), "%Y-%b-%d %H:%M")

    def calcDensitySA(self):
        # calculate water density
        sp = gsw.conversions.SP_from_C(self.df.cond, self.df.temp, self.df.sea_pressure) # calculate practical sal from conductivity
        sa = gsw.conversions.SA_from_SP(sp, self.df.sea_pressure, lon=np.nanmean(self.df.lon), lat=np.nanmean(self.df.lat)) # calculate absolute salinity from practical sal and lat/lon
        pt = gsw.conversions.pt0_from_t(sa, self.df.temp, self.df.sea_pressure) # calculate potential temperature from salinity and temp
        ct = gsw.conversions.CT_from_pt(sa, pt) # calculate critical temperature from potential temperature
        rho = gsw.density.rho(sa, ct, self.df.sea_pressure) # calculate density from absolute sal, critical temp, and pressure
        sigma = gsw.sigma0(sa, ct)

        self.df['density'] = rho
        self.df['absolute_salinity'] = sa
        self.df['potential_temperature'] = pt
        self.df['sigma'] = sigma
        self.df['conservative_temperature'] = ct

        self.df = self.df.query('absolute_salinity > 0')

    def calcW(self):
        prof_inds = self.df.profile_index.values
        t = np.float64(self.df.time.values)

        for val in np.unique(prof_inds)[0:len(np.unique(prof_inds))-1]:
            if val != np.float64('nan'):
                start = np.where(prof_inds==val)[0][0]
                end = np.where(prof_inds==val)[0][-1]

                d_sub = self.df.depth_m.values[start:end]
                d_filt = d_sub > 1
                d_sub = d_sub[d_filt]
                d_sub = d_sub.reshape(-1,1)

                t_sub = t[start:start+len(d_sub)].reshape(-1,1)

                model = LinearRegression()
                model.fit(t_sub, d_sub)
                d_pred = model.predict(t_sub)
                w = -1*model.coef_[0][0]*np.power(10,9) # convert to m/s from m/ns

    def makeDataDisplayStrings(self):
        keys = self.data_strings.keys()
        for key in keys:
            sub_keys = self.data_strings[key].keys()
            for sub_key in sub_keys:
                var_vals = self._data[key][sub_key]
                var_mean = np.nanmean(self._data[key][sub_key])
                var_max = np.max(self._data[key][sub_key])
                var_min = np.min(self._data[key][sub_key])

                low_thresh = np.min(self.acceptable_ranges[key][sub_key])
                high_thresh = np.max(self.acceptable_ranges[key][sub_key])
                if var_mean > low_thresh and var_mean < high_thresh:
                    self.data_strings[key][sub_key].append(f"{sub_key}: {np.nanmean(self._data[key][sub_key]):0.3f} {self.units[sub_key]}")
                    self.data_strings[key][sub_key].append(14)
                    self.data_strings[key][sub_key].append('k')
                    self.data_strings[key][sub_key].append('normal')
                    self.data_strings[key][sub_key].append('normal')

                else:
                    self.data_strings[key][sub_key].append(f"{sub_key}: {np.nanmean(self._data[key][sub_key]):0.3f} {self.units[sub_key]}")
                    self.data_strings[key][sub_key].append(14)
                    self.data_strings[key][sub_key].append('r')
                    self.data_strings[key][sub_key].append('italic')
                    self.data_strings[key][sub_key].append('bold')
                
                if var_min < low_thresh or var_max > high_thresh:
                    if key == "dive":
                        self.problem_dives.append(f"{self._data[key][sub_key].index(var_max) + 1}")
                    else:
                        self.problem_dives.append(f"{self._data[key][sub_key].index(var_max) + 2}")

    def makeFlightPanel(self): # needs to be pointed to a save directory
        fig = plt.figure(constrained_layout = True, figsize=(11, 8.5))
        gs = fig.add_gridspec(6, 7)
        ax1 = fig.add_subplot(gs[0:3, :5])
        ax2 = fig.add_subplot(gs[4, :5], sharex=ax1)
        ax3 = fig.add_subplot(gs[5, :5], sharex=ax2)
        ax4 = fig.add_subplot(gs[2:, 5:], fc='wheat', alpha=0.5)
        ax5 = fig.add_subplot(gs[:2, 5:], fc='wheat', alpha=0.5)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        for label in ax3.get_xticklabels(which='major'):
            label.set(rotation=15, horizontalalignment='center')

        ax1.xaxis.set_tick_params(labelbottom=False)
        ax2.xaxis.set_tick_params(labelbottom=False)
        ax4.xaxis.set_tick_params(labelbottom=False)
        ax4.yaxis.set_tick_params(labelleft=False)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.xaxis.set_tick_params(labelbottom=False)
        ax5.yaxis.set_tick_params(labelleft=False)
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax1.set_title(f"{self.glider} profiles {np.unique(self.df.profile_index)[0]} - {np.unique(self.df.profile_index)[-2]}")
        ax1.invert_yaxis()
        ax1.set_ylabel("Depth [m]")
        ax2.set_ylabel("Pitch [deg]")
        ax3.set_ylabel("Roll [deg]")
        ax3.set_xlabel("Time")

        d_plot = ax1.scatter(self.df.time, self.df.depth_m, c=self.df.density,s=1.5)
        ax2.plot(self.df.time, self.df.pitch_deg)
        ax3.plot(self.df.time, self.df.roll_deg)
        cax = fig.add_axes((4/7-1/2, 11/24, 15/24, .05))
        fig.colorbar(d_plot, cax=cax, orientation='horizontal', label='Density [kg $\\bullet m^{-3}$]')

        _keys = self.data_strings.keys()
        for i, key in enumerate(_keys):
            sub_keys = self.data_strings[key].keys()
            ax4.text(0.03, 0.98-i*0.5, f"Avg {key}", verticalalignment='top', fontsize=15, c = 'k')

            for j, sub_key in enumerate(sub_keys):
                ax4.text(0.08, 0.93-i*0.5-j*0.05, f"{self.data_strings[key][sub_key][0]}", verticalalignment='top', 
                         fontsize=self.data_strings[key][sub_key][1], c = self.data_strings[key][sub_key][2], 
                         fontstyle=self.data_strings[key][sub_key][3], fontweight=self.data_strings[key][sub_key][4])

        # ax4.text(0.03, 0.98, "Avg dive", verticalalignment='top', c = 'k')
        ax5.text(0.03, 0.95, f"Bat %: {(1-(np.max(self.df.amphr)/self.max_amphrs))*100:0.2f}\
                 \n\nAmphr: {np.max(self.df.amphr):0.2f}\
                 \n\n# dives: {len(np.unique(self.df.profile_index))//2}\
                 \n\nProblem profiles: \n{np.unique(self.problem_dives)}", 
                 verticalalignment='top', c = 'k', fontsize=16)
        
        if self.data_dir == "data/processed/":
            plt.savefig(f"images/toSend/{self.glider}_flight_panel_full_time_series.png")
        else:
            plt.savefig(f"images/toSend/{self.glider}_flight_panel_{self.date}.png")
        # plt.show()
    
    def makeSciDpPanel(self):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(11, 4), sharey=True)
        axs[0].invert_yaxis()
        axs[0].set_ylabel("Depth [m]")

        profs = self.profiles_to_make.keys()
        colors = ['k', 'c0', 'r', 'g']
        ind_dfs = {}
        dfs_of_same_columns = {}

        for i, prof in enumerate(self.profiles_to_make.keys()):
            axs[i].xaxis.tick_top()
            for j, var in enumerate(self.profiles_to_make[prof]):
                axs[i].scatter(self.df[var], self.df.depth_m, s=5, label=var)
                axs[i].legend()

        if self.data_dir == "data/processed/":
            plt.savefig(f"images/toSend/{self.glider}_depth_profiles_full_time_series.png")
        else:
            plt.savefig(f"images/toSend/{self.glider}_depth_profiles_{self.date}.png")
        # plt.show()

    def makeSciTSPanel(self):
        # code adapted from Jacob Partida
        for var in self.sci_vars:
            fig = plt.figure(constrained_layout = True, figsize=(15, 8.5))
            gs = fig.add_gridspec(6, 7)
            ax1 = fig.add_subplot(gs[0:, 0:3])
            ax2 = fig.add_subplot(gs[0:3, 3:])
            # ax4 = fig.add_subplot(gs[4:, 3:])

            ax2.invert_yaxis()
            # ax3.invert_yaxis()
            # ax4.invert_yaxis()

            ax1.set_xlabel("Salinity [$g \\bullet kg^{-1}$]", fontsize=14)
            ax1.set_ylabel("Temperature [°C]", fontsize=14)


            s_lims = (np.floor(data.df.absolute_salinity.min()-0.5),
            np.ceil(data.df.absolute_salinity.max()+0.5))
            t_lims = (np.floor(data.df.conservative_temperature.min()-0.5),
                    np.ceil(data.df.conservative_temperature.max()+0.5))
            S = np.arange(s_lims[0],s_lims[1]+0.1,0.1)
            T = np.arange(t_lims[0],t_lims[1]+0.1,0.1)
            Tg, Sg = np.meshgrid(T,S)
            sigma = gsw.sigma0(Sg,Tg)

            c0 = ax1.contour(Sg, Tg, sigma, colors='grey', zorder=1)
            c0l = plt.clabel(c0, colors='k', fontsize=9)

            p0 = ax1.scatter(self.df.absolute_salinity, self.df.conservative_temperature, 
                             c=self.df[var], cmap=self.sci_colors[var], s=5)
            cbar0 = fig.colorbar(p0, label=f"{var} [{self.units[var]}]", location='left')

            if "cat" in self.data_dir:
                ax2.scatter(self.df[var], self.df.depth_m, s=2)
                ax2.set_xlabel(f"{self.df[var]} [{self.units[var]}]", fontsize=14)
                ax2.set_ylabel("Depth [m]", fontsize=14)
            else:
                ax2.set_xlabel("Time", fontsize=14)
                ax2.set_ylabel("Depth [m]", fontsize=14)
                ax2.scatter(self.df.time, self.df.depth_m, c=self.df[var], cmap=self.sci_colors[var], s=2)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                for label in ax2.get_xticklabels(which='major'):
                    label.set(rotation=15, horizontalalignment='center')
            
            ax3 = fig.add_subplot(gs[3:, 3:])
            ax3.set_xlabel('\n\n\nLongitude [Deg]', fontsize=14)
            ax3.set_ylabel('Latitude [Deg]\n\n\n', fontsize=14)
            glider_lon, glider_lat = self.df.lon, self.df.lat
            glider_lon_min = np.min(self.df.lon)
            glider_lon_max = np.max(self.df.lon)
            glider_lat_min = np.min(self.df.lat)
            glider_lat_max = np.max(self.df.lat)
            glider_lon_mean = np.nanmean(glider_lon)
            glider_lat_mean = np.nanmean(glider_lat)
            
            lon_range = glider_lon_max-glider_lon_min
            lat_range = glider_lat_max-glider_lat_min

            if f"{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}" in os.listdir("mapPickles/"):
                with open(f"mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}", "rb") as fd:
                    map = pickle.load(fd)
            else:
                map = Basemap(llcrnrlon=glider_lon_min-.05, llcrnrlat=glider_lat_min-.01,
                            urcrnrlon=glider_lon_max+.01, urcrnrlat=glider_lat_max+.05, resolution='f') # create map object
                with open(f"mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}", "wb") as fd:
                    pickle.dump(map, fd, protocol=-1)

            map.drawcoastlines()
            map.drawcountries()
            # map.bluemarble()
            map.fillcontinents('#e0b479')
            map.drawlsmask(ocean_color = "#7bcbe3", resolution='f')
            map.drawparallels(np.linspace(glider_lat_min-0.1, glider_lat_max+0.1, 5), labels=[1,0,0,1], fmt="%0.2f")
            map.drawmeridians(np.linspace(glider_lon_min-0.1, glider_lon_max+0.1, 5), labels=[1,0,0,1], fmt="%0.3f", rotation=20)
            x, y = map(glider_lon, glider_lat)
            map.scatter(x, y, c=self.df[var], s=5, cmap=self.sci_colors[var])
            map.scatter(x.iloc[-1], y.iloc[-1], c='red', s=50, marker="*")


            axins = zoomed_inset_axes(ax3, 0.008, loc='upper left')
            axins.set_xlim(glider_lon_min-3, glider_lon_max+3)
            axins.set_ylim(glider_lat_min-3, glider_lat_max+3)

            if f"{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}_inset" in os.listdir("mapPickles/"):
                with open(f"mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}_inset", "rb") as fd:
                    map_in = pickle.load(fd)
            else:
                map_in = Basemap(llcrnrlon=glider_lon_min-3, llcrnrlat=glider_lat_min-3,
                            urcrnrlon=glider_lon_max+3, urcrnrlat=glider_lat_max+3, resolution='f') # create map object
                with open(f"mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}_inset", "wb") as fd:
                    pickle.dump(map_in, fd, protocol=-1)

            map_in.drawcoastlines()
            map_in.drawcountries()
            map_in.fillcontinents('#e0b479')

            # map.drawparallels(np.linspace(glider_lat_min-3, glider_lat_max+3, 5), labels=[1,0,0,1], fmt="%0.2f")
            # map.drawmeridians(np.linspace(glider_lon_min-3, glider_lon_max+3, 5), labels=[1,0,0,1], fmt="%0.3f", rotation=20)
            # map.bluemarble()
            map_in.drawlsmask(ocean_color = "#7bcbe3", resolution='f')
            x, y = map_in(glider_lon_mean, glider_lat_mean)
            map_in.scatter(x, y, c='r', s=5)
            map_in.scatter(x, y, c='r', s=100, alpha=0.25)
            # mark_inset(ax3, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
            if self.data_dir == "data/processed/":
                plt.savefig(f"images/toSend/{self.glider}_{var}_ts_panel_full_time_series.png")
            else:
                plt.savefig(f"images/toSend/{self.glider}_{var}_ts_panel_{self.date}.png")

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')   
        # ax.scatter(self.df.lon, self.df.lat, self.df.depth_m, c=self.df.cdom)
        # ax.invert_zaxis()
        # plt.show()
    
    def makeSegmentedDf(self):
        prof_inds = self.df.profile_index.values
        depth = self.df.depth_m.values
        t = np.float64(self.df.time.values)
        time = self.df.time.values
        roll = self.df.roll_deg.values
        pitch = self.df.pitch_deg.values
        amphr = self.df.amphr.values
        vac = self.df.vacuum.values

        sub_keys = ["w", "pitch", "roll", "dt", "amphr", "vac"]
        dive_bool = np.zeros(len(self.df.time))

        for val in np.unique(prof_inds)[0:len(np.unique(prof_inds))-1]: # fill above 'dive_segs' dict with values
            if val != np.float64('nan'): 
                start = np.where(prof_inds==val)[0][0]
                end = np.where(prof_inds==val)[0][-1]

                d_sub = depth[start:end]
                d_filt = d_sub > 1
                d_sub = d_sub[d_filt]
                d_sub = d_sub.reshape(-1,1)

                t_sub = t[start:start+len(d_sub)].reshape(-1,1)
                _t = t[start:start+len(d_sub)]
                dt = np.max(_t) - np.min(_t); dt = dt/10**9/3600
                r_mean = np.nanmean(roll[start:end])
                p_mean = np.nanmean(pitch[start:end])
                a =  amphr[start:end]
                da = np.max(a) - np.min(a)
                _vac = vac[start:start+len(d_sub)]
                vac_mean = np.mean(_vac)

                model = LinearRegression()
                model.fit(t_sub, d_sub)
                d_pred = model.predict(t_sub)
                w = -1*model.coef_[0][0]*np.power(10,9) # convert to m/s from m/ns

                p_sub_raw = pitch[start:end]
                _vars = {"w":w, "pitch":p_mean, "roll":r_mean, "dt":dt, "amphr":da, "vac": vac_mean}

                if w > 0:
                    parent_key = "climb"
                    for i in range(start, end):
                        dive_bool[i] = -1
                else:
                    parent_key = "dive"
                    for i in range(start, end):
                        dive_bool[i] = 1

                for sub_key in sub_keys:
                    self._data[parent_key][sub_key].append(float(_vars[sub_key]))

        self.df["dive_bool"] = dive_bool
        self.df = self.df.query("dive_bool > 0")
        # self.df = self.df.query("cond > 0")
    
    def moveDataFilesToProcessed(self):
        prev_calls = os.listdir('data/processed/')
        prev_calls.remove('.DS_Store')

        for file in os.listdir("data/new_data"):
            os.rename(f"data/new_data/{file}", f"data/processed/{file}")

    def reset(self):
        self.tm = 0
        self.depth = 0
        self.roll = 0
        self.pitch = 0
        self.oil_vol = 0
        self.pressure = 0
        self.temp = 0
        self.cond = 0
        self.lat = 0
        self.lon = 0
        self.sea_pressure = 0
        self.vacuum = -1
        self.amphr = 0
        self.o2 = 0
        self.cdom = 0
        self.chlor = 0
        self.backscatter = 0

        self.data_strings = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : ("w = ### m/s", 14, 'r')}}

        self._data = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : [list of values, e.g., roll]}}      
        self.data_dir = "data/processed/"  

    def makeFullDeploymentPlots(self):
        self.readRaw()
        self.makeDf()
        self.getProfiles()
        self.calcDensitySA()
        self.calcW()
        self.makeSegmentedDf()
        self.makeDataDisplayStrings()
        self.makeFlightPanel()
        self.makeSciTSPanel()
        self.makeSciDpPanel()
        self.saveDataCsv()
    
    def sendEmail(self):
        image_dir = "images/toSend/"
        csv_dir = "data/toSend/csv.zip"
        email_doer = doEmail(image_dir, csv_dir, self.glider, self.date, self.n_yos, self.n_yos_tot, 
                             self.dive_start, self.dive_end, self.dep_start, self.dep_end)
        email_doer.send()

    def moveImages(self):
        im_dirs = os.listdir("images/")
        if "sent" not in im_dirs: os.mkdir("images/sent/")
        ims_to_move = os.listdir('images/toSend/')
        ims_to_move.remove('.DS_Store')

        for file in ims_to_move:
            os.rename(f"images/toSend/{file}", f"images/sent/{file}")
    
    def packageTimeSeries(Self):
        if "timeseries" not in os.listdir("images/"):
            os.mkdir(f"images/timeseries/")

        for file in os.listdir("images/toSend/"):
            if "time" in file:
                os.rename("images/toSend/"+ file, "images/timeseries/" + file)

        with zipfile.ZipFile('images/timeseries.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            zipdir('images/timeseries/', zipf)

        for file in os.listdir("images/timeseries/"): os.remove("images/timeseries/"+file)
        os.removedirs("images/timeseries/")
        os.rename("images/timeseries.zip", "images/toSend/timeseries.zip")

    def saveDataCsv(self):
        if "new" in self.data_dir:
            self.df.to_csv(f"data/toSend/csv/mostRecentScrape/{self.glider}_{self.date}_recent_scrape_data.csv")
        else:
            self.df.to_csv(f"data/toSend/csv/timeseries/{self.glider}_{self.date}_timeseries.csv")

    def zipAndDelCsv(self):
        with zipfile.ZipFile('data/toSend/csv.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            zipdir('data/toSend/csv/', zipf)

        dirs = os.listdir("data/toSend/csv/")
        dirs.remove('.DS_Store')
        for dir in dirs: 
            for file in  os.listdir("data/toSend/csv/"+dir):
                os.remove("data/toSend/csv/"+dir+"/"+file)
    
    def run(self):
        self.getWorkingDirs()
        self.readRaw()
        self.makeDf()
        self.moveDataFilesToProcessed()
        self.getProfiles()
        self.calcDensitySA()
        self.calcW()
        self.makeSegmentedDf()
        self.makeDataDisplayStrings()
        self.makeFlightPanel()
        self.makeSciTSPanel()
        self.makeSciDpPanel()
        self.saveDataCsv()
        self.reset()
        self.makeFullDeploymentPlots()
        self.packageTimeSeries()
        self.zipAndDelCsv()
        self.sendEmail()
        self.moveImages()


class doEmail:
    def __init__(self, image_dir, csv_dir, glider, date, yos, yos_tot, dive_start, dive_end, dep_start, dep_end):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.glider_name = glider
        self.date = date
        self.yos = yos
        self.yos_total = yos_tot

        self.subject = f"{self.glider_name} science plots on {self.date}"

        self.body = f"This is an automated email and is a prototype for the automated webscrapping application.\nThese data were scraped from SFMC on {self.date} for the glider named \"{self.glider_name}\".\
            \n\n{self.glider_name} performed {self.yos+1:0.0f} half-yos from {dive_start} to {dive_end} prior to the last scrape.\
            \n{self.glider_name} has performed a total of {self.yos_total+1:0.0f} from {dep_start} to {dep_end}.\
            \n\nThe full deployment time series can be downloaded in the attached zipfile.\n\nPlease do not reply to this email, as caleb does not know how to write code to handle that...\
            \n\nFor data questions or concerns, please email caleb.flaim@noaa.gov and sam.woodman@noaa.gov."

        self.sender_email = "esdgliders@gmail.com"
        self.recipiants = ["caleb.flaim@noaa.gov", "esdgliders@gmail.com"] #nmfs.swfsc.esd-gliders@noaa.gov , "jacob.partida@noaa.gov", 
                        #    "jen.walsh@noaa.gov", "anthony.cossio@noaa.gov", "christian.reiss@noaa.gov",
                        #    "eric.bjorkstedt@noaa.gov"
        self.password =   # access_secret_version('ggn-nmfs-usamlr-dev-7b99', 'esdgliders-email')input("Type your password and press enter:")
    
    def send(self):
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.recipiants)
        message["Subject"] = self.subject
        message["Bcc"] = "esdgliders@gmail.com"  # Recommended for mass emails

        # Add body to email
        message.attach(MIMEText(self.body, "plain"))
        foo = os.listdir(self.image_dir)
        foo.remove(".DS_Store")
        for file in foo:
            with open(self.image_dir+"/"+file, "rb") as attachment:
                # Add file as application/octet-stream
                # Email client can usually download this automatically as attachment
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

                # Encode file in ASCII characters to send by email    
                encoders.encode_base64(part)

                # Add header as key/value pair to attachment part
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {file}",
                )

            # Add attachment to message and convert message to string
            message.attach(part)
            text = message.as_string()

        with open(self.csv_dir, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email    
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                "attachment; filename= csv.zip",
            )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()

        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.recipiants, text)

    

if __name__ == "__main__":
    data = gliderData()
    data.run()

