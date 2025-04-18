# Data modules
import numpy as np
import pandas as pd
import xarray as xr 
import dbdreader
import matplotlib.pyplot as plt
from matplotlib import colormaps 
import matplotlib.tri as tri
import os
import sys
import shutil
import argparse
import scipy as sp
from scipy import odr
from scipy.signal import argrelextrema
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
# import pyglider.utils as pgu
import esdglider.gcp as gcp
import logging

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

#need to access gcp secret for email pw

class gliderData:
    def __init__(self, args):
        if args.logfile == "":
            logging.basicConfig(
            format='%(asctime)s %(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
            level=getattr(logging, args.loglevel.upper()), 
            datefmt="%Y-%m-%d %H:%M:%S")
        else:
            logging.basicConfig(
            filename=args.logfile,
            filemode="a",
            format='%(asctime)s %(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
            level=getattr(logging, args.loglevel.upper()), 
            datefmt="%Y-%m-%d %H:%M:%S")

        # Glider info
        self.deployment = args.deployment
        self.glider = self.deployment.split("-")[0]
        self.project = args.project
        self.max_amphrs = args.amphours
        self.new_data_bool = False
        self.year = args.year

        # Data path info
        gcp_bucket_dep_date = self.deployment.split("-")[1][0:4]
        gcp_bucket_dep_date.strip()
        self.processed_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/"
        self.data_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/new_data/" # set data dir specific to glider
        self.data_parent_dir = "/opt/slocumRtDataVisTool/data/" # set parent dir for data folders (any glider)
        # self.cache_dir = "/opt/standard-glider-files/Cache/" # set directoy path for cache
        self.cache_dir = "/mnt/gcs/deployment/cache/" # set directoy path for cache
        self.gcp_mnt_bucket_dir = f"/mnt/gcs/deployment/{self.project}/{self.year}/{self.deployment}/data/binary/rt/"
        self.image_dir = f"/opt/slocumRtDataVisTool/images/{self.glider}/"
        self.image_parent_dir = f"/opt/slocumRtDataVisTool/images/"
        self.goto_dir = self.data_parent_dir+f"{self.glider}/goto/"
        self.date = ""
        
        # Eng data defaults
        self.n_yos = -1
        self.n_yos_tot = -1
        self.dive_start = -1
        self.dive_end = -1
        self.dep_start = -1
        self.dep_end = -1

        # Sci data defaults
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
        
        self.profiles_to_make = {"profile3":["cdom"],
                                 "profile0":["absolute_salinity"],
                                 "profile1":["temp"],
                                 "profile2":["sigma"],
                                 "profile4":["oxygen"],
                                 "profile5":["chlorophyl"]} # Dictionary to be looped thru to make depth profiles of scienec vars
      
        self.sci_vars = ["cdom", "chlorophyl", "oxygen", "backscatter"] # List of science variables
        self.sci_colors = {"cdom":cmo.solar, "chlorophyl": cmo.algae, "oxygen":cmo.tempo, "backscatter":colormaps['terrain'], 
                            "conservative_temperature":cmo.thermal, "absolute_salinity":cmo.haline, "density":colormaps['cividis']} # dictionary of colormaps to be used for listed science variables
        
        self.list_sci_vars = ['backscatter', 'chlorophyl', 'cdom', 'oxygen','temp', 'cond','density', 'absolute_salinity', 'potential_temperature', 'sigma', 'conservative_temperature']
        self.cross_section_sci_vars = ['backscatter', 'chlorophyl', 'cdom', 'oxygen','density', 'absolute_salinity', 'conservative_temperature']
        # Dictionary used to store strings of processed data stats and 
        # parameters to display the string (e.g., color, condasize)
        self.data_strings = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : ("w = ### m/s", 14, 'r')}}
        
        # Empty dictionary to store data for dives/climbs
        self._data = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : [list of values, e.g., roll]}}
        
        # Dictionary of units for plotting purposes
        self.units = {"w":'m/s', "pitch":'deg', "density":"kg $\\bullet m^{-3}$","roll":'deg', "dt":'hrs', 
        "amphr":'Ahr', "vac":'inHg', "cdom":"ppb", "chlorophyl":"µg•$L^{-l}$", "oxygen":"mL•$L^{-1}$", 
        "backscatter":"$m^{-1}$", "absolute_salinity":"g•$kg^{-1}$", "conservative_temperature":"°C", "temp":"°C", "sigma":"kg $\\bullet m^{-3}$"} # 
        # Dictionary to store acceptable ranges fpr flight parameters
        self.acceptable_ranges = {"dive":{"w":[-0.08, -0.20], "pitch":[-20, -29], "roll":[-5, 5], "dt":[0.25, 3], "amphr":[0.05, 1.0], "vac":[6, 10]},
                                  "climb":{"w":[0.08, 0.20], "pitch":[20, 29], "roll":[-5, 5], "dt":[0.25, 3], "amphr":[0.05, 1.0], "vac":[6, 10]}}
    

    def checkGliderDataDir(self):
        logging.info("Checking for existing glider directory.")

        if self.glider not in os.listdir(self.data_parent_dir):
            logging.info(f"Making new data directory for {self.glider}")

            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}")
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/new_data/")
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/")
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/mostRecentScrape/")
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/timeseries/")

        if "new_data" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/new_data/")
        if "goto" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}"):
            os.makedirs(self.goto_dir)
        if "processed" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/")
        if "toSend" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/")
        if "csv" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv")
        if "mostRecentScrape" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/mostRecentScrape/")
        if "timeseries" not in os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/"):
            os.makedirs(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/timeseries/")


        self.data_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/new_data/"

        if self.glider not in os.listdir(self.image_parent_dir):
            logging.info(f"Making new image directory for {self.glider}")
            os.makedirs(self.image_dir)
        if "sent" not in os.listdir(self.image_dir):
            os.makedirs(self.image_dir+"sent/")
        if "toSend" not in os.listdir(self.image_dir):
            os.makedirs(self.image_dir+"toSend/")

    def checkNewData(self):
        logging.info("Checking for new s/tbd files in mounted bucket.")

        files_in_bucket = set(os.listdir(self.gcp_mnt_bucket_dir))
        files_in_processed = set(os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/"))
        new_files = list(files_in_bucket - files_in_processed)

        # copy not-processed files fom the mounted bucket to data/new_data
        if len(new_files)>0:
            logging.info(f"New files found! Moving to {self.data_dir}")
            self.new_data_bool = True
            for file in new_files:
                shutil.copy(self.gcp_mnt_bucket_dir+file, self.data_dir)

        else:
            logging.info("No new data found. Sending email.")
            self.sendNoData("No new data was found for processing.")
            logging.info("No data email sent.")

    def sendNoData(self, msg):
        image_dir = f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/"
        csv_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv.zip"
        email_doer = doEmail(image_dir, csv_dir, self.glider, self.date, self.n_yos, self.n_yos_tot, 
                             self.dive_start, self.dive_end, self.dep_start, self.dep_end, msg)

        email_doer.sendNoData()

    def readRaw(self):
        # interp_fact_keys = ["m_depth", 'sci_flbbcd_bb_units','sci_flbbcd_chlor_units', 'sci_flbbcd_cdom_units','sci_oxy4_oxygen','m_coulomb_amphr_total', 'm_vacuum','m_roll', 'm_pitch', 'm_de_oil_vol', 'sci_water_pressure', 'sci_water_temp', 'sci_water_cond', 'm_lat', 'm_lon']
        # interp_fact = {}
        # for key in interp_fact_keys:
        #     interp_fact[key] = returnNan
        try:
        # Creates dbd reader object
            logging.info("Reading in raw binary files")
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
            logging.info("Binary files successfully read in.")
        except Exception as e:
            logging.warning(e, exc_info=True)
            self.sendNoData(e)

    def makeDf(self): 
        logging.info("Making pandas dataframe.")
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

        logging.info("Filtering bogus values.")
        # NOTE: should do somehting here to check for gaps in the data to avoid 
        self.df = self.df.query("cond > 0 & cond < 60")
        self.df = self.df.query("temp > -2 & temp < 100")
        self.df = self.df.query("sea_pressure > -1")
        self.df = self.df.query("chlorophyl > 0")
        # self.df.loc[self.df['chlorophyl'] == 0] = np.NaN
        # self.df.loc[self.df['oxygen'] == 0] = np.NaN
        # self.df = self.df.dropna()
        self.df.reindex()
        logging.info("Dataframe made successfully!")

    def getProfiles(self, min_dp=5.0, filt_time=150, profile_min_time=300):
        # code in this function was modified from pyglider to work with pandas dataframe
        
        if 'pressure' not in self.df.columns:
            logging.warning('No "pressure" variable in the data set; not searching for profiles')

        profile = self.df.pressure.values * 0
        direction = self.df.pressure.values * 0
        pronum = -1

        good = np.where(np.isfinite(self.df.pressure))[0]
        dt = float(np.median(
            np.diff(self.df.time.values[good[:200000]]).astype(np.float64)) * 1e-9)
        logging.info(f'dt, {dt}')
        filt_length = int(filt_time / dt)

        min_nsamples = int(profile_min_time / dt)
        logging.info('Filt Len  %d, dt %f, min_n %d', filt_length, dt, min_nsamples)
        if filt_length > 1:
            p = np.convolve(self.df.pressure.values[good],
                            np.ones(filt_length) / filt_length, 'same')
        else:
            p = self.df.pressure.values[good]
        decim = int(filt_length / 3)
        if decim < 2:
            decim = 2
        # why?  because argrelextrema doesn't like repeated values, so smooth
        # then decimate to get fewer values:
        pp = p[::decim]
        maxs = argrelextrema(pp, np.greater)[0]
        mins = argrelextrema(pp, np.less)[0]
        mins = good[mins * decim]
        maxs = good[maxs * decim]
        if mins[0] > maxs[0]:
            mins = np.concatenate(([0], mins))
        if mins[-1] < maxs[-1]:
            mins = np.concatenate((mins, good[[-1]]))

        logging.debug(f'mins: {len(mins)} {mins} , maxs: {len(maxs)} {maxs}')

        pronum = 0
        p = self.df.pressure.to_numpy()
        nmin = 0
        nmax = 0

        _num_iters = 0
        while (nmin < len(mins)) and (nmax < len(maxs)):
            _num_iters += 1
            # try:
            nmax = np.where(maxs > mins[nmin])[0]
            if len(nmax) >= 1:
                nmax = nmax[0]
            else:
                break
            logging.debug(nmax)
            ins = range(int(mins[nmin]), int(maxs[nmax]+1))
            # logging.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
            # logging.debug(f'Down, {ins}, {p[ins[0]].values},{p[ins[-1]].values}')
            logging.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
            logging.debug(f'Down, {ins}, {p[ins[0]]},{p[ins[-1]]}')
            if ((len(ins) > min_nsamples) and
                    (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
                profile[ins] = pronum
                direction[ins] = +1
                pronum += 1
            nmin = np.where(mins > maxs[nmax])[0]
            if len(nmin) >= 1:
                nmin = nmin[0]
            else:
                break
            ins = range(maxs[nmax], mins[nmin])
            # logging.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
            # logging.debug(f'Up, {ins}, {p[ins[0]].values}, {p[ins[-1]].values}')
            logging.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
            logging.debug(f'Up, {ins}, {p[ins[0]]}, {p[ins[-1]]}')
            if ((len(ins) > min_nsamples) and
                    (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
                # up
                profile[ins] = pronum
                direction[ins] = -1
                pronum += 1
            
            # except KeyError:
            #     print("errored")
            #     pass

        logging.info(f"Found profiles in {_num_iters} iterations.")
        self.df['profile_index'] = profile
        self.df['profile_direction'] = direction

        logging.info("Profiles added to dataframe.")
        if "new" in self.data_dir:
            self.n_yos = np.max(self.df.profile_index)
            self.dive_start = datetime.strftime(np.min(self.df.time), "%Y-%b-%d %H:%M")
            self.dive_end = datetime.strftime(np.max(self.df.time), "%Y-%b-%d %H:%M")
        else:
            self.n_yos_tot = np.max(self.df.profile_index)
            self.dep_start = datetime.strftime(np.min(self.df.time), "%Y-%b-%d %H:%M")
            self.dep_end = datetime.strftime(np.max(self.df.time), "%Y-%b-%d %H:%M")
        
    def calcDensitySA(self):
        logging.info("Calculating SA, rho, CT.")
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

        logging.info("Filtering bogus values.")
        self.df = self.df.query('absolute_salinity > 0 & absolute_salinity < 60')
        self.df = self.df.query('conservative_temperature > -2 & conservative_temperature < 50')
        # self.df = self.df.dropna()
        #nan_num = np.float64('nan')
        #self.df = self.df.query('(conservative_temperature != @nan_num) & (absolute_salinity != @nan_num)')
        logging.info("SA, rhp, CT added to data frame.")

    def calcW(self, t_sub, d_sub, val):
        model = LinearRegression()
        model.fit(t_sub, d_sub)
        d_pred = model.predict(t_sub)
        w = -1*model.coef_[0][0]*np.power(10,9) # convert to m/s from m/ns

        logging.info(f"Vertical speed for profile{val} is {w} m/s.")
        return w

    def makeDataDisplayStrings(self):
        logging.info("Making display strings.")
        keys = self.data_strings.keys()
        for key in keys:
            logging.info(f"Making display strings for {key}")
            sub_keys = self.data_strings[key].keys()
            for sub_key in sub_keys:
                if len(self._data[key][sub_key])>0:
                    logging.info(f"Making {key} {sub_key} display string.")
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
        logging.info("Display strings finished successfully.")

    def makeFlightPanel(self): # needs to be pointed to a save directory
        try:
            logging.info("Making flight data panel.")
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

            ax1.set_title(f"{self.glider} profiles {np.unique(self.df.profile_index)[0]} - {np.unique(self.df.profile_index)[-1]}")
            ax1.invert_yaxis()
            ax1.set_ylabel("Depth [m]")
            ax2.set_ylabel("Pitch [deg]")
            ax3.set_ylabel("Roll [deg]")
            ax3.set_xlabel("Time")

            d_plot = ax1.scatter(self.df.time, self.df.depth_m, c=self.df.density, cmap=self.sci_colors['density'], s=1.5)
            ax2.plot(self.df.time, self.df.pitch_deg)
            ax3.plot(self.df.time, self.df.roll_deg)
            cax = fig.add_axes((4/7-1/2, 11/24, 15/24, .05))
            fig.colorbar(d_plot, cax=cax, orientation='horizontal', label='Density [kg $\\bullet m^{-3}$]')

            _keys = self.data_strings.keys()
            for i, key in enumerate(_keys):
                try:
                    sub_keys = self.data_strings[key].keys()
                    logging.info(f"Setting text values for {key}")
                    ax4.text(0.03, 0.98-i*0.5, f"Avg {key}", verticalalignment='top', fontsize=15, c = 'k')

                    for j, sub_key in enumerate(sub_keys):
                        logging.info(f"Setting text values for {sub_key} on iteration i, j = {i}, {j}.")
                        ax4.text(0.08, 0.93-i*0.5-j*0.05, f"{self.data_strings[key][sub_key][0]}", verticalalignment='top', 
                                fontsize=self.data_strings[key][sub_key][1], c = self.data_strings[key][sub_key][2], 
                                fontstyle=self.data_strings[key][sub_key][3], fontweight=self.data_strings[key][sub_key][4])
                except IndexError as err:
                    logging.warning(f"Flight data {key} {sub_key} had {err} error.")
            # ax4.text(0.03, 0.98, "Avg dive", verticalalignment='top', c = 'k')
            ax5.text(0.03, 0.95, f"Bat %: {(1-(np.max(self.df.amphr)/self.max_amphrs))*100:0.2f}\
                    \n\nAmphr: {np.max(self.df.amphr):0.2f}\
                    \n\n# dives: {len(np.unique(self.df.profile_index))//2}\
                    \n\nProblem profiles: \n{np.unique(self.problem_dives)}", 
                    verticalalignment='top', c = 'k', fontsize=16)
            
            if self.data_dir == f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/":
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_flight_panel_full_time_series.png")
                logging.info("Saved flight panel for timeseries.")
            else:
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_flight_panel_{self.date}.png")
                logging.info("Saved flight panel for new data.")
            
            logging.info("Flight panel finished successfully.")
            # plt.show()

        except Exception as e:
            logging.warning(e, exc_info=True)
            pass
    
    def makeSciDpPanel(self):
        try:
            logging.info("Making depth profiles.")
            fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(14, 4), sharey=True)
            axs[0].invert_yaxis()
            axs[0].set_ylabel("Depth [m]")
            # x1 = axs[1].twiny()
            # axs[1].xaxis.tick_top()
            # x2 = axs[1].twiny()
            # x3 = axs[1].twiny()


            # x_list = [axs[1], x2, x3]
            # cs = ['#1f77b4', '#ff7f0e','#2ca02c']
            
            # axs[1].tick_params(axis='x', colors='#1f77b4')
            # x2.tick_params(axis='x', colors='#ff7f0e')
            # x3.tick_params(axis='x', colors='#2ca02c')
            # x3.spines['bottom'].set_position(('outward', 50))
            
            profs = self.profiles_to_make.keys()
            # colors = ['k', 'c0', 'r', 'g']
            ind_dfs = {}
            dfs_of_same_columns = {}

            for i, prof in enumerate(self.profiles_to_make.keys()):
                axs[i].xaxis.tick_top()
                for j, var in enumerate(self.profiles_to_make[prof]):
                    logging.info(f"Making {var} plot {prof}.")
                    axs[i].xaxis.set_label_position('top') 
                    axs[i].scatter(self.df[var], self.df.depth_m, s=5, label=var)
                    axs[i].set_xlabel(f"{var} [{self.units[var]}]")
                    # axs[i].legend()

            if self.data_dir == f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/":
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_depth_profiles_full_time_series.png")
                logging.info("Saved depth profiles for timeseries.")
            else:
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_depth_profiles_{self.date}.png")
                logging.info("Saved depth profiles for new data.")
            # plt.show()

        except Exception as e:
            logging.warning(e, exc_info=True)
            pass
    
    def makeSciTSPanel(self):
        try:
            logging.info("Makinf TS plots.")
            # code adapted from Jacob Partida
            s_lims = (np.floor(np.min(self.df.absolute_salinity)-0.5),
            np.ceil(np.max(self.df.absolute_salinity)+0.5))
            logging.info(f"s_lims: {s_lims}")

            t_lims = (np.floor(np.min(self.df.conservative_temperature)-0.5),
                        np.ceil(np.max(self.df.conservative_temperature)+0.5))
            logging.info(f"t_lims: {t_lims}")
            # print(t_lims)
            S = np.arange(s_lims[0],s_lims[1]+0.1,0.1)
            T = np.arange(t_lims[0],t_lims[1]+0.1,0.1)
            Tg, Sg = np.meshgrid(T,S)
            sigma = gsw.sigma0(Sg,Tg)

            goto_parser = gotoParser(self.goto_dir+"goto_l10.ma")
            wpts = goto_parser.parse()

            map_lon_border = 0.1
            map_lat_border = 0.5

            for var in self.sci_vars:
                try:
                    logging.info(f"Making {var} TS plot")
                    fig = plt.figure(constrained_layout = True, figsize=(9, 15))
                    gs = fig.add_gridspec(12, 6)

                    ax0 = fig.add_subplot(gs[0:3,:]) # timeseries
                    ax1 = fig.add_subplot(gs[3:7, 0:4]) # TS
                    ax2 = fig.add_subplot(gs[3:7:, 4:]) # Depth profile

                    ax0.invert_yaxis()
                    ax2.invert_yaxis()
                    ax2.xaxis.set_label_position('top') 

                    ax1.set_xlabel("Salinity [$g \\bullet kg^{-1}$]", fontsize=14)
                    ax1.set_ylabel("Temperature [°C]", fontsize=14)
                    
                    ax0.set_xlabel("Time", fontsize=14)
                    ax0.set_ylabel("Depth [m]", fontsize=14)

                    p0= ax0.scatter(self.df.time, self.df.depth_m, c=self.df[var], cmap=self.sci_colors[var], s=3)
                    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    for label in ax0.get_xticklabels(which='major'):
                        label.set(rotation=15, horizontalalignment='center')

                    c0 = ax1.contour(Sg, Tg, sigma, colors='grey', zorder=1)
                    c0l = plt.clabel(c0, colors='k', fontsize=9)

                    ax1.scatter(self.df.absolute_salinity, self.df.conservative_temperature, 
                                    c=self.df[var], cmap=self.sci_colors[var], s=5)
                    cbar0 = fig.colorbar(p0, label=f"{var} [{self.units[var]}]", location='bottom', shrink=1)

                    ax2.scatter(self.df[var], self.df.depth_m, s=2)
                    ax2.set_xlabel(f"{var} [{self.units[var]}]", fontsize=14)
                    ax2.set_ylabel("Depth [m]", fontsize=14)

                    glider_lon, glider_lat = wpts.wpt_lon.values, wpts.wpt_lat.values #self.df.lon, self.df.lat
                    glider_lon_min = np.min(glider_lon)
                    glider_lon_max = np.max(glider_lon)
                    glider_lat_min = np.min(glider_lat)
                    glider_lat_max = np.max(glider_lat)
                    glider_lon_mean = np.nanmean(glider_lon)
                    glider_lat_mean = np.nanmean(glider_lat)
                    
                    # lon_range = glider_lon_max-glider_lon_min
                    # lat_range = glider_lat_max-glider_lat_min
                    ax3 = fig.add_subplot(gs[7:, :]) # map
                    ax3.set_xlabel('\n\n\nLongitude [Deg]', fontsize=14)
                    ax3.set_ylabel('Latitude [Deg]\n\n\n', fontsize=14)

                    # if f"{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}" in os.listdir("/opt/slocumRtDataVisTool/mapPickles/"):
                    #     with open(f"/opt/slocumRtDataVisTool/mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}", "rb") as fd:
                    #         m = pickle.load(fd)
                    # else:
                    m = Basemap(llcrnrlon=glider_lon_min-map_lon_border, llcrnrlat=glider_lat_min-map_lat_border,
                                urcrnrlon=glider_lon_max+map_lon_border, urcrnrlat=glider_lat_max+map_lat_border, resolution='f', ax=ax3) # create map object
                        # with open(f"/opt/slocumRtDataVisTool/mapPickles/{self.glider}_{glider_lon_mean:0.0f}_{glider_lat_mean:0.0f}", "wb") as fd:
                        #     pickle.dump(m, fd, protocol=-1)
                    
                    wpt_lon, wpt_lat = m(wpts.wpt_lon.values, wpts.wpt_lat.values)

                    m.drawcoastlines()
                    m.drawcountries()
                    # map.bluemarble()
                    m.fillcontinents('#e0b479')
                    m.drawlsmask(ocean_color = "#7bcbe3", resolution='f')
                    m.drawparallels(np.linspace(glider_lat_min-map_lat_border, glider_lat_max+map_lat_border, 5), labels=[1,0,0,1], fmt="%0.2f")
                    m.drawmeridians(np.linspace(glider_lon_min-map_lon_border, glider_lon_max+map_lon_border, 5), labels=[1,0,0,1], fmt="%0.3f", rotation=20)

                    m.plot(wpt_lon, wpt_lat, c="#808080", linestyle="--", marker='o')
                    x, y = m(self.df.lon, self.df.lat)
                    # map.scatter(x, y, c=self.df[var], s=5, cmap=self.sci_colors[var])
                    m.scatter(x, y, c='k', s=5, zorder=2.5)
                    # m.scatter(x, y, c=self.df[var], s=5, zorder=2.5, cmap=self.sci_colors[var])
                    m.scatter(x.iloc[-1], y.iloc[-1], c='red', s=75, marker="*", zorder=2.5)
                    

                    if self.data_dir == f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/":
                        plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_{var}_ts_panel_full_time_series.png")
                        logging.info("Saved TS plots for timeseries.")
                    else:
                        plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_{var}_ts_panel_{self.date}.png")
                        logging.info(f"Saved TS plots for {var} new data.")

                    plt.close()
                    logging.info(f"Cleared {var} Figure.")

                except ZeroDivisionError as e:
                    logging.warning(f"{var} TS plot experienced {e}. Try updating map pickles.")
                    continue

        except Exception as e:
            logging.warning(e, exc_info=True)
            pass

    def makeSpatialCrossSections(self):
 
        logging.info("Making spatial cross sections.")
        df = self.df.copy()
        df = df.drop_duplicates()
        df = df.dropna()
        fig, axs = plt.subplots(7,2, sharey=True, figsize=(11,25))
        axs_left = axs[:,0]
        axs_right = axs[:,1]
        # axs[0,0].invert_yaxis()
        
        try:
            for i, var in enumerate(self.cross_section_sci_vars):
                good = np.where(np.isfinite(df[var]))[0]
                lon = np.asarray(df.lon.values[good]).flatten()
                lat = np.asarray(df.lat.values[good]).flatten()
                depth = np.asarray(df.depth_m.values[good]).flatten()
                sci_var = np.asarray(df[var].values[good]).flatten()

                _triang = tri.Triangulation(lon, depth)
                axs_left[i].tripcolor(_triang, sci_var, cmap=self.sci_colors[var], shading='gouraud')
                axs_left[i].scatter(lon, depth, c=sci_var, cmap=self.sci_colors[var], alpha=1, s=1.5)
                axs_left[i].set_xlabel("Deg Longitiude", fontsize=14)
                axs_left[i].set_ylabel("Depth [m]", fontsize=14)
                axs_left[i].invert_yaxis()
                axs_left[i].tick_params(labelsize=12)

                for label in axs_left[i].get_xticklabels(which='major'):
                    label.set(rotation=15, horizontalalignment='center')

                _triang = tri.Triangulation(lat, depth)
                plot = axs_right[i].tripcolor(_triang, sci_var, cmap=self.sci_colors[var], shading='gouraud')
                axs_right[i].scatter(lat, depth, c=sci_var, cmap=self.sci_colors[var], alpha=1, s=1.5)
                axs_right[i].set_xlabel("Deg Latitude", fontsize=14)
                axs_right[i].tick_params(labelsize=12)
                for label in axs_right[i].get_xticklabels(which='major'):
                    label.set(rotation=15, horizontalalignment='center')

                cbar = fig.colorbar(plot, ax=axs_right[i],location='right')
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label(label= f"{var} [{self.units[var]}]", fontsize=14)

                cbar.ax.yaxis.set_tick_params(labelrotation=15)
                logging.info(f"Finished cross sections for {var}")

            # plt.tight_layout()
            if self.data_dir == f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/":
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_spatial_sections_time_series_{self.date}.png")
            else:
                plt.savefig(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{self.glider}_spatial_sections_{self.date}.png")

            logging.info("Saved cross section panel.")

        except Exception as e:
            logging.warning(e, exc_info=True)
            pass

    def makeSegmentedDf(self):
        logging.info("Segmenting dataframe.")
        logging.info("Calculating vertical speeds.")
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
        double_u = np.zeros(len(self.df.time))

        for val in np.unique(prof_inds)[0:len(np.unique(prof_inds))-1]: # fill above 'dive_segs' dict with values
            logging.info(f"Dataframe segmented for profile {val}")
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

                w = self.calcW(t_sub, d_sub, val)
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
                
                for i in range(start, end):
                    double_u[i] = w
                for sub_key in sub_keys:
                    self._data[parent_key][sub_key].append(float(_vars[sub_key]))
                    
        logging.info("Vertical velocities calculated successfuly.")
        logging.info("Filtering dataframe for climbs")
        self.df["dive_bool"] = dive_bool
        self.df["W"] = double_u
        # self.df = self.df.query("dive_bool > 0")
        self.df.loc[self.df['dive_bool'] < 0, self.list_sci_vars] = np.NaN
        self.df.loc[self.df['profile_direction'] < 0, self.list_sci_vars] = np.NaN
        self.df.dropna()
        logging.info("Climbs removed.")
        # self.df = self.df.query("cond > 0")
    
    def moveDataFilesToProcessed(self):
        prev_calls = os.listdir(self.processed_dir)
        if '.DS_Store' in prev_calls: prev_calls.remove('.DS_Store')

        for file in os.listdir(self.data_dir):
            os.rename(f"{self.data_dir}{file}", f"{self.processed_dir}{file}")
            logging.info(f"Moved {self.data_dir}{file} to {self.processed_dir}{file}.")
        logging.info(f"Datafiles moved to /data/{self.glider}/processed.")

    def reset(self):
        logging.info("Reseting to make timeseries plots.")
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
        self.data_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/processed/"  
        logging.info("gliderData has been reset.")

    def makeFullDeploymentPlots(self):
        logging.info("Making full deployment plots.")
        # self.checkGliderDataDir()
        self.readRaw()
        self.makeDf()
        self.getProfiles()
        self.calcDensitySA()
        self.makeSpatialCrossSections()
        self.makeSegmentedDf()
        self.makeDataDisplayStrings()
        self.makeFlightPanel()
        self.makeSciTSPanel()
        self.makeSciDpPanel()
        self.saveDataCsv()
        logging.info("Full deployment plots made and saved.")
    
    def sendEmail(self):
        logging.info("Attaching plots/data and sending email.")
        image_dir = f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/"
        csv_dir = f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv.zip"
        email_doer = doEmail(image_dir, csv_dir, self.glider, self.date, self.n_yos, self.n_yos_tot, 
                             self.dive_start, self.dive_end, self.dep_start, self.dep_end)
        email_doer.send()
        logging.info("Email sent.")

    def moveImages(self):
        im_dirs = os.listdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/")
        if "sent" not in im_dirs: os.mkdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/sent/")
        ims_to_move = os.listdir(f'/opt/slocumRtDataVisTool/images/{self.glider}/toSend/')
        if '.DS_Store' in ims_to_move: ims_to_move.remove('.DS_Store')

        for file in ims_to_move:
            os.rename(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/{file}", f"/opt/slocumRtDataVisTool/images/{self.glider}/sent/{file}")
        logging.info("Images moved to sent.")

    def packageTimeSeries(self):
        if "timeseries" not in os.listdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/"):
            logging.info("Making timeseries folder.")
            os.mkdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/")

        for file in os.listdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/"):
            if "time" in file:
                os.rename(f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/"+ file, f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/" + file)

        logging.info("Zipping timeseries plots.")
        with zipfile.ZipFile(f'/opt/slocumRtDataVisTool/images/{self.glider}/timeseries.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            zipdir(f'/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/', zipf)

        for file in os.listdir(f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/"): os.remove(f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/"+file)
        os.removedirs(f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries/")
        os.rename(f"/opt/slocumRtDataVisTool/images/{self.glider}/timeseries.zip", f"/opt/slocumRtDataVisTool/images/{self.glider}/toSend/timeseries.zip")
        logging.info("Timeseries packaged successfully.")

    def saveDataCsv(self):
        if "new" in self.data_dir:
            self.df.to_csv(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/mostRecentScrape/{self.glider}_{self.date}_recent_scrape_data.csv")
            logging.info("New data saved into CSV file.")
        else:
            self.df.to_csv(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/timeseries/{self.glider}_{self.date}_timeseries.csv")
            logging.info("Timeseries saved into CSV file.")

    def zipAndDelCsv(self):
        with zipfile.ZipFile(f'/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            zipdir(f'/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/', zipf)
        logging.info("New and timeseries csv data zipped")

        dirs = os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/")
        if '.DS_Store' in dirs: dirs.remove('.DS_Store')
        for dir in dirs: 
            for file in  os.listdir(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/"+dir):
                os.remove(f"/opt/slocumRtDataVisTool/data/{self.glider}/toSend/csv/"+dir+"/"+file)
        logging.info("Unzipped CSV data deleted.")

    def run(self):
        logging.info(f"\n\n\nRunning gliderData for {self.glider}.\n")
        self.checkGliderDataDir()
        self.checkNewData()
        if self.new_data_bool:
            logging.info("New data found: making plots.")
            self.readRaw()
            self.makeDf()
            self.moveDataFilesToProcessed()
            self.getProfiles()
            self.calcDensitySA()
            self.makeSegmentedDf()
            self.makeDataDisplayStrings()
            self.makeSciTSPanel()
            self.makeSpatialCrossSections()
            self.makeFlightPanel()
            self.makeSciDpPanel()
            self.saveDataCsv()
            self.reset()
            self.makeFullDeploymentPlots()
            self.packageTimeSeries()
            self.zipAndDelCsv()
            self.sendEmail()
            self.moveImages()
            logging.info("Plots generated and sent. Code done.")
        else: logging.info("No new data found. Code done.")

class gotoParser:
    def __init__(self, goto_dir):
        self.goto_dir = goto_dir
        self.wpts = []
    
    def parse(self):
        bool_found_pts = False
        with open(self.goto_dir) as fn:
            for line in fn:
                if "<start:waypoints>" in line:
                    bool_found_pts = True
                    continue
                
                if bool_found_pts and "<end:waypoints>" not in line:
                    self.wpts.append(line.strip().split())

        coords = self.convertCoords()
        return coords

    def convertCoords(self):
        self.wpt_lats = []
        self.wpt_lons = []

        for set in self.wpts:
            for i, pt in enumerate(set):
                dec_loc = pt.index('.')
                deg = float(pt[0:dec_loc-2])
                dec_min = float(pt[dec_loc-2:])/60

                if pt[0]=='-':
                    dec_degs = deg - dec_min
                else:
                    dec_degs = deg + dec_min

                if i == 0:
                    self.wpt_lons.append(dec_degs)
                else:
                    self.wpt_lats.append(dec_degs)

        bar = {'wpt_lon':self.wpt_lons, "wpt_lat":self.wpt_lats}
        foo = pd.DataFrame(bar)

        return foo

class doEmail:
    def __init__(self, image_dir, csv_dir, glider, date, yos, yos_tot, dive_start, dive_end, dep_start, dep_end, msg=None):
        self.msg = str(msg)
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.glider_name = glider
        self.date = date
        self.yos = yos
        self.yos_total = yos_tot

        self.subject = f"{self.glider_name} science plots on {self.date}"
        self.no_data_subject = f"New no data scraped for {self.glider_name}"

        self.body = f"This is an automated email and is a prototype for the real-time data visualization application.\nThese data were scraped from SFMC on {self.date} for the glider named \"{self.glider_name}\".\
            \n\n{self.glider_name} performed {self.yos+1:0.0f} half-yos from {dive_start} to {dive_end} prior to the last scrape.\
            \n{self.glider_name} has performed a total of {self.yos_total+1:0.0f} from {dep_start} to {dep_end}.\
            \n\nThe full deployment time series images can be downloaded in the attached zipfile.\nCSV data of the most recent scrape and full timeseries can be downloaded from the csv.zip file.\
            \nPlease note that the data have not been quality checked and have only been filtered for blatantly wrong values.\
            \n\nPlease do not reply to this email, as caleb does not know how to write code to handle that...\
            \n\nFor data questions/concerns/suggestions, please email caleb.flaim@noaa.gov and sam.woodman@noaa.gov."
        
        # self.no_data_body = "No new data was found for processing."

        self.sender_email = "esdgliders@gmail.com"
        self.no_data_recipiants = ["caleb.flaim@noaa.gov", "esdgliders@gmail.com"]
        #self.recipiants = ["caleb.flaim@noaa.gov", "esdgliders@gmail.com", "jacob.partida@noaa.gov", "jen.walsh@noaa.gov", "anthony.cossio@noaa.gov", "christian.reiss@noaa.gov","eric.bjorkstedt@noaa.gov", "jason.c.clark@noaa.gov", "george.watters@noaa.gov", "jefferson.hinke@noaa.gov", "douglas.krause@noaa.gov"]
        self.recipiants = ["caleb.flaim@noaa.gov", "esdgliders@gmail.com"] #nmfs.swfsc.esd-gliders@noaa.gov , "jacob.partida@noaa.gov", 
                        #    "jen.walsh@noaa.gov", "anthony.cossio@noaa.gov", "christian.reiss@noaa.gov",
                        #    "eric.bjorkstedt@noaa.gov"
        self.password = # to fill in on VM  # access_secret_version('ggn-nmfs-usamlr-dev-7b99', 'esdgliders-email')input("Type your password and press enter:")
    
    def sendNoData(self):
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.no_data_recipiants)
        message["Subject"] = self.no_data_subject
        message["Bcc"] = "esdgliders@gmail.com"  # Recommended for mass emails
        message.attach(MIMEText(self.msg, "plain"))

        text = message.as_string()
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.recipiants, text)

    def send(self):
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.recipiants)
        message["Subject"] = self.subject
        message["Bcc"] = "esdgliders@gmail.com"  # Recommended for mass emails

        # Add body to email
        message.attach(MIMEText(self.body, "plain"))
        foo = os.listdir(self.image_dir)
        if ".DS_Store" in foo: foo.remove(".DS_Store")
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
    # gcp.gcs_mount_bucket("amlr-gliders-deployments-dev", "/mnt/deployments/")
    arg_parser = argparse.ArgumentParser(description=gliderData.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployment',
        type=str,
        help='Deployment name, eg amlr03-20220425')

    arg_parser.add_argument('project', 
        type=str,
        help='Glider project name', 
        choices=['FREEBYRD', 'REFOCUS', 'SANDIEGO', 'ECOSWIM'])

    arg_parser.add_argument('-l', '--loglevel',
        type=str,
        help='Verbosity level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info')
    
    arg_parser.add_argument('--logfile',
        type=str,
        help='File to which to write logs',
        default='')

    arg_parser.add_argument('year',
        type=str,
        help='e.g., year the glider was recovered.')

    arg_parser.add_argument('amphours',
        type=int,
        help='e.g., 300 Ahrs, 800 Ahrs')

    try:
        parsed_args = arg_parser.parse_args()
        data = gliderData(parsed_args)
        data.run()

    except Exception as e:
        if parsed_args.logfile == "":
            logging.basicConfig(
            format='%(asctime)s %(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
            level=getattr(logging, parsed_args.loglevel.upper()), 
            datefmt="%Y-%m-%d %H:%M:%S")
        else:
            logging.basicConfig(
            filename=parsed_args.logfile,
            filemode="a",
            format='%(asctime)s %(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
            level=getattr(logging, parsed_args.loglevel.upper()), 
            datefmt="%Y-%m-%d %H:%M:%S")

        logging.warning(e, exc_info=True)

# [x] The code needs to start by mounting the amlr-dev-#### bucket to /mnt/ using esdglider.gcs_mount_bucket()
# [x] The code them needs to take in a variable (e.g., $DEPLOYMENT) and find the glider name using the String.split('-') function --> first element is the glider name.
# [] Will need to implement argparse to be able to take in the deployment name and access GCP secrets to send the email.
# [] Need to check for existing gliderName directory in /data/ --> if the directory exists, do nothing. If the directory doesn't exist, make the directory and add "new_data" and "processed" subdirectories. 
# [] Need to change the data file paths to work with f"opt/slocumRtDataVisTooldata/{args.glider}/new_data | processed" where the glider folder is determined by the shell script call to run rtGliderPlots.py.
# [] Upon mounting the GCP bucket, the code needs to check what files in the bucket it has already processed and which are the new files --> during a deployment, already processed files will be stored on the VM in /opt/slocumRtDataVisTool/data/gliderName/procressed. The code will compare the lists of files in the bucket and locally and copy the remaining files to /opt/slocumRtDataVisTool/data/gliderName/new_data to be processed.
# [] The code should now run perfectly with no issues whatsoever.