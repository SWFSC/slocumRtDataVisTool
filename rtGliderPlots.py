import numpy as np
import pandas as pd
import xarray as xr 
import dbdreader
import matplotlib.pyplot as plt
import os
import scipy as sp
from sklearn.linear_model import LinearRegression
import gsw
import matplotlib.dates as mdates
import cmocean.cm as cmo

class gliderData:
    def __init__(self):
        self.data_dir = ""
        self.cache_dir = ""
        self.glider = ""
        self.max_amphrs = 300

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
        self.data_strings = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : ("w = ### m/s", 14, 'r')}}
        
        self._data = {"dive":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]},
                      "climb":{"w":[], "pitch":[], "roll":[], "dt":[], "amphr":[], "vac":[]}} # {"key" : {"sub_key" : [list of values, e.g., roll]}}
        self.units = {"w":'m/s', "pitch":'deg', "roll":'deg', "dt":'hrs', "amphr":'Ahr', "vac":'inHg'}

        
    def getWorkingDirs(self):
        # data_dir = input("Enter the absolute file path to the binary files for your deployment (e.g., /Users/NAME/Documents/etc.): ")
        self.data_dir = "/Users/cflaim/Documents/data/georgeSbDeployment/george-from-glider-20240913T150418/"
        # cache_dir = (input("If your cahce files are in a separate directy than the data files, enter the absolute path to them. Other wise press enter.") or "0")
        self.cache_dir = "/Users/cflaim/Documents/GitHub/standard-glider-files/Cache/"

        glider = os.listdir(self.data_dir)
        for g in glider:
            if ".tbd" in g:
                self.glider = g.split("-")[0]
                break

    def inputDiveParams(self):
        pass

    def readRaw(self): # needs to be pointed to working directory
        if self.cache_dir == "0":
            dbd = dbdreader.MultiDBD(pattern=self.data_dir+'george*.[st]bd')
        else:
            dbd = dbdreader.MultiDBD(pattern=self.data_dir+'george*.[st]bd', cacheDir=self.cache_dir)

        self.tm, self.depth, self.chlor, self.cdom, self.o2, self.amphr, self.vacuum, self.roll, self.pitch, self.oil_vol, self.pressure, self.temp, self.cond, self.lat, self.lon = dbd.get_sync("m_depth", 'sci_flbbcd_chlor_units', 'sci_flbbcd_cdom_units','sci_oxy4_oxygen','m_coulomb_amphr','m_vacuum','m_roll', 'm_pitch', 'm_de_oil_vol', 
                                'sci_water_pressure', 'sci_water_temp', 'sci_water_cond', 'm_lat', 'm_lon')
        
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

    def makeDf(self): # look at Sam's code for example
        _dt = np.array([dbdreader.dbdreader.epochToDateTimeStr(t, timeformat='%H:%M:%S') for t in self.tm])
        time = np.array([t[0]+ ' ' +t[1] for t in _dt])

        vars = [self.tm, self.depth, self.chlor, self.cdom, self.o2, self.amphr, self.vacuum, self.roll, self.pitch, self.oil_vol, self.sea_pressure, self.pressure, self.temp, 
                self.cond, self.lat, self.lon] # variable values for pandas DF
        columns = ['time', 'depth_m', 'chlorophyl', 'cdom', 'oxygen', 'amphr', 'vacuum', 'roll_deg', 'pitch_deg', 'oil_vol','sea_pressure', 'pressure', 
                'temp', 'cond', 'lat', 'lon'] # column names for pandas df
        
        df_dict = {} # empty dict to be filled 
        # fill above dictionary
        for i, var in enumerate(columns):
            df_dict[var] = vars[i]

        self.df = pd.DataFrame(df_dict) # make pandas df from dict
        self.df['time'] = pd.to_datetime(self.df['time']) # convert the time column to DateTime type from datestrings
    
    def getProfiles(self):
        """
        Adds profile index and direction to existing Pandas DataFrame made from slocum binary data files.
        """
        min_dp=10.0
        inversion=3.
        filt_length=7
        min_nsamples=14

        profile = self.df.pressure.values * np.nan
        direction = self.df.pressure.values * np.nan
        pronum = 1
        lastpronum = 0

        good = np.where(~np.isnan(self.df.pressure))[0]
        p = np.convolve(self.df.pressure.values[good],
                        np.ones(filt_length) / filt_length, 'same')
        dpall = np.diff(p)
        inflect = np.where(dpall[:-1] * dpall[1:] < 0)[0]
        for n, i in enumerate(inflect[:-1]):
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

        self.df['profile_index'] = profile
        self.df['profile_direction'] = direction

    def calcDensitySA(self):
        # calculate water density
        sp = gsw.conversions.SP_from_C(self.cond, self.temp, self.sea_pressure) # calculate practical sal from conductivity
        sa = gsw.conversions.SA_from_SP(sp, self.sea_pressure, lon=np.nanmean(self.lon), lat=np.nanmean(self.lat)) # calculate absolute salinity from practical sal and lat/lon
        pt = gsw.conversions.pt0_from_t(sa, self.temp, self.sea_pressure) # calculate potential temperature from salinity and temp
        ct = gsw.conversions.CT_from_pt(sa, pt) # calculate critical temperature from potential temperature
        rho = gsw.density.rho(sa, ct, self.sea_pressure) # calculate density from absolute sal, critical temp, and pressure
        sigma = gsw.sigma0(sa, ct)

        self.df['density'] = rho
        self.df['absolute_salinity'] = sa
        self.df['potential_temperature'] = pt
        self.df['sigma'] = sigma
        self.df['conservative_temperature'] = ct

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
                self.data_strings[key][sub_key].append(f"{sub_key}: {np.nanmean(self._data[key][sub_key]):0.3f} {self.units[sub_key]}")
                self.data_strings[key][sub_key].append(14)
                self.data_strings[key][sub_key].append('k')

    def makeFlightPlots(self): # needs to be pointed to a save directory
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
                         fontsize=self.data_strings[key][sub_key][1], c = self.data_strings[key][sub_key][2])

        # ax4.text(0.03, 0.98, "Avg dive", verticalalignment='top', c = 'k')
        ax5.text(0.03, 0.95, f"Bat %: {100-np.max(self.df.amphr)/self.max_amphrs:0.2f}\
                 \n\nAmphr: {np.max(self.df.amphr):0.2f}\
                 \n\n# dives: {len(np.unique(self.df.profile_index))//2}", 
                 verticalalignment='top', c = 'k', fontsize=16)

        plt.show()
    
    def makeSciPlots(self):
        # code adapted from Jacob Partida
        fig = plt.figure(constrained_layout = True, figsize=(15, 8.5))
        gs = fig.add_gridspec(6, 7)
        ax1 = fig.add_subplot(gs[0:, 0:3])
        ax2 = fig.add_subplot(gs[0:2, 3:])
        ax3 = fig.add_subplot(gs[2:4, 3:])
        ax4 = fig.add_subplot(gs[4:, 3:])

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
        p0 = ax1.scatter(self.df.absolute_salinity, self.df.conservative_temperature, c=self.df.oxygen, cmap=cmo.tempo)
        cbar0 = fig.colorbar(p0, label="Oxygen [$mL \\bullet L^{-1}$]")

        plt.show()
    
    def makeSegmentedDf(self): # make this the data string function????
        prof_inds = self.df.profile_index.values
        depth = self.df.depth_m.values
        t = np.float64(self.df.time.values)
        time = self.df.time.values
        roll = self.df.roll_deg.values
        pitch = self.df.pitch_deg.values
        amphr = self.df.amphr.values
        vac = self.df.vacuum.values

        sub_keys = ["w", "pitch", "roll", "dt", "amphr", "vac"]

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
                else:
                    parent_key = "dive"

                for sub_key in sub_keys:
                    self._data[parent_key][sub_key].append(float(_vars[sub_key]))

    def run(self):
        self.getWorkingDirs()
        self.readRaw()
        self.makeDf()
        self.getProfiles()
        self.calcDensitySA()
        self.calcW()
        self.makeSegmentedDf()
        self.makeDataDisplayStrings()
        # self.makeFlightPlots()
        self.makeSciPlots()


if __name__ == "__main__":
    data = gliderData()
    data.run()
# data = gliderData()
# data.run()
