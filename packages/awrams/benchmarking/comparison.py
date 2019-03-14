import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import awrams.utils.datetools as dt
from awrams.utils.metatypes import ObjectDict,New

from .stats import build_stats_df,standard_percentiles
from .utils import valid_only,infer_freq,resample_to_months_df,resample_to_years_df
from .model import Selector
from awrams.utils import config_manager

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('comparison')

SAMPLE_RATE = {'d':1, 'daily':1,
               'm':2, 'monthly':2,
               'y':3, 'yearly':3, 'annually':3}

system_profile = config_manager.get_system_profile().get_settings()

FIG_SIZE = system_profile['BENCHMARKING']['FIG_SIZE']

class ComparisonSet(object):
    def __init__(self,obs_df,ref_name,var_name,period,how=np.mean,annual_rule='A'):
        self.period = period
        self.var_name = var_name
        self.ref_name = ref_name
        self.aggr_how = how
        self.annual_rule = annual_rule

        obs_df = obs_df.loc[(obs_df.index >= period[0])&(obs_df.index <= period[-1])]
        self.obs = New()
        self.obs.data = obs_df
        self.obs.freq = infer_freq(obs_df)
        if self.obs.freq != 'y':
            self.obs.monthly = self.obs.data.resample(rule='m', how=self.aggr_how)
        self.obs.annual = self.obs.data.resample(rule='A', how=self.aggr_how)
#        self.obs.freq = infer_freq(obs_df)
        self.models = ObjectDict()
        self._cmap = plt.get_cmap('gist_rainbow')
        self.selection = Selector([])

    def _assign_colours(self):
        n = len(self.models)
        n = 10 if n < 10 else n
        m_colours = np.linspace(0.,1.,n)
        for i, m in enumerate(self.models.values()):
            m.colour = np.array(self._cmap(m_colours[i])) * np.array((0.95,0.75,0.9,1.0))

    def _intersect(self, mod, obs):
        _mod = {}
        _obs = {}

        for site in self.obs.data.columns:
            try:
                if obs[site] is None or mod[site] is None:
                    #logger.warning("no data for sitemissing site in model dataframe...skipping :%s",site)
                    continue

                valid_isect_idx = valid_only(obs[site]).index.intersection(valid_only(mod[site]).index)
                _mod[site] = mod[site].loc[valid_isect_idx]
                _obs[site] = obs[site].loc[valid_isect_idx]
            except KeyError:
                pass
                #logger.warning("missing site in model dataframe...skipping :%s",site)
        return _mod, _obs

    def _add_model(self,model_df,name,freq='d'):
        self.selection._add(name)
        m = ObjectDict(freq=SAMPLE_RATE[freq])
        self.models[name] = m
        m.name = name

        m.data = ObjectDict()
        m.obs  = ObjectDict()

        m.data.raw = model_df.loc[self.period]

        if self.obs.freq != 'y':
            if freq == 'd' and self.obs.freq == 'd':
                m.data.daily,m.obs.daily = self._intersect(m.data.raw, self.obs.data)
                m.data.monthly = resample_to_months_df(m.data.daily, self.aggr_how)
                m.obs.monthly  = resample_to_months_df(m.obs.daily, self.aggr_how)

            elif freq == 'm' or self.obs.freq == 'm':
                if freq == 'm':
                    _mod = m.data.raw.resample(rule='m', how=self.aggr_how)
                else: # assume must be daily
                    _mod = resample_to_months_df(m.data.raw, self.aggr_how)
                if self.obs.freq == 'm':
                    _obs = self.obs.data.resample(rule='m', how=self.aggr_how)
                else:
                    _obs = resample_to_months_df(self.obs.data, self.aggr_how)

                m.data.monthly,m.obs.monthly = self._intersect(_mod,_obs)

            else:
                raise Exception('model freq is %s' % repr(freq))

            m.data.annual = resample_to_years_df(m.data.monthly, self.aggr_how, min_months=6)
            m.obs.annual = resample_to_years_df(m.obs.monthly, self.aggr_how, min_months=6)
            m.data.annual,m.obs.annual = self._intersect(m.data.annual,m.obs.annual)

        else: #obs are annual (recharge)
            m.data.annual = m.data.raw.resample(rule='a', how=self.aggr_how)
            m.obs.annual = self.obs.data

        m.stats = ObjectDict(freq=freq)
        m.stats.daily = None
        m.stats.monthly = None
        if freq == 'd' and self.obs.freq == 'd':
            m.stats.daily = build_stats_df(m.obs.daily, m.data.daily, m.obs.daily.keys())
        if self.obs.freq != 'y':
            m.stats.monthly = build_stats_df(m.obs.monthly, m.data.monthly, m.obs.monthly.keys())
        if m.obs.annual is not None and m.data.annual is not None:
            m.stats.annual = build_stats_df(m.obs.annual, m.data.annual, m.obs.annual.keys())

        self.build_objfunc_stats(m.stats)
        self._assign_colours()

    def build_objfunc_stats(self, stats):
        if stats.daily is not None:
            for site in stats.daily.columns:
                try:
                    stats.daily.loc['fobj',site] = (stats.monthly.loc['nse',site] + stats.daily.loc['nse',site])/2. - 5*(np.abs(np.log(1+stats.daily.loc['bias_relative',site])))**2.5
                except KeyError:
                    pass
        if stats.monthly is not None:
            for site in stats.monthly.columns:
                try:
                    stats.monthly.loc['fobj',site] = stats.monthly.loc['nse',site] - 5*(np.abs(np.log(1+stats.monthly.loc['bias_relative',site])))**2.5
                except KeyError:
                    pass
        for site in stats.annual.columns:
            try:
                stats.annual.loc['fobj',site] = stats.annual.loc['nse',site] - 5*(np.abs(np.log(1+stats.annual.loc['bias_relative',site])))**2.5
            except KeyError:
                pass

    def _iter_models(self, freq):
        def miter():
            for name in self.selection():
                m = self.models[name]
                if m.freq <= SAMPLE_RATE[freq]:
                    yield m
        return miter()

    def _get_ax(self,kwargs):
        if 'ax' in kwargs:
            ax = kwargs['ax']
            del kwargs['ax']
        else:
            plt.figure(figsize=FIG_SIZE)
            ax = plt.subplot(1,1,1)
        return ax

    def plot_timeseries(self,site,freq='m',model=None,**kwargs):
        '''
        Plot timeseries of data at the specified site and frequency
        '''
        from functools import partial

        ax = self._get_ax(kwargs)

        # def _plot(ax,series,label,colour):
        def _plot(series,label,colour):
            #+++ fix for pandas 0.16.1 legend label bug (see https://github.com/pydata/pandas/issues/10119)
            series.name = label
            # series.plot(legend=True,axes=ax,color=colour)
            series.plot(legend=True,color=colour)
        # plot = partial(_plot,ax=ax)
        plot = partial(_plot)

        if freq == 'raw':
            #self.obs.data[site].plot(legend=True,axes=ax,color='black',label=self.ref_name)
            plot(series=self.obs.data[site],label=self.ref_name,colour='black')
            for name in self.selection():
                m = self.models[name]
                #m.data.raw[site].plot(legend=True,axes=ax,color=m.colour,label=m.name)
                plot(series=m.data.raw[site],label=m.name,colour=m.colour)
        else:
            tf = dt.validate_timeframe(freq).lower()
            _freq = freq == 'y' and 'A' or freq
 
            if model is not None:       
                if not model in self.models:
                    logger.critical("%s not found in %s",model,self.models)
                    return None
                else:
                    plot(series=self.models[model].obs[tf][site].resample(_freq).asfreq(),label=self.ref_name,colour='black')
                    plot(series=self.models[model].data[tf][site].resample(_freq).asfreq(),label=self.models[model].name,colour=self.models[model].colour)
            else:
                if freq == 'd':
                    plot(series=self.obs.data[site].resample(_freq).asfreq(),label=self.ref_name,colour='black')
                elif freq == 'm':
                    plot(series=self.obs.monthly[site].resample(_freq).asfreq(),label=self.ref_name,colour='black')
                elif freq == 'y':
                    plot(series=self.obs.annual[site].resample(_freq).asfreq(),label=self.ref_name,colour='black')

                for m in self._iter_models(freq):
                    try:
                        plot(series=m.data[tf][site].resample(_freq).asfreq(),label=m.name,colour=m.colour)
                    except:
                        logger.warning("no data to plot for %s site %s",m.name,site)

        ax.legend(loc='best')
        ax.set_title("%s" % site)
        ax.set_ylabel(self.var_name)
        ax.set(**kwargs)
        ax.grid()
        return ax

    def plot_cdf(self,statistic='pearsons_r',freq='m', **kwargs):
        '''
        Plot the empirical CDF for the specified statistic and frequency
        '''
        tf = dt.validate_timeframe(freq).lower()

        ax = self._get_ax(kwargs)

        for m in self._iter_models(freq):
            y = sorted(m.stats[tf].loc[statistic, m.stats[tf].columns != 'all'].dropna())  # temporary fix for broken cdf's
            ax.plot(np.linspace(0,1.,len(y)),y,color=m.colour,label=m.name)

        ax.set_xlabel("Catchments below (%)")
        ax.set_ylabel(statistic)
        ax.legend(loc='best')
        ax.set(**kwargs)
        ax.grid()
        return ax

    def plot_box(self,statistic,freq='m', **kwargs):
        '''
        Show a box-plot for the specified statistic and timeframe
        '''
        tf = dt.validate_timeframe(freq).lower()

        ax = self._get_ax(kwargs)

        data = []
        colours = []
        names = []

        for m in self._iter_models(freq):
            data.append(m.stats[tf].loc[statistic, m.stats[tf].columns != 'all'])
            colours.append(m.colour)
            names.append(m.name)

        box = ax.boxplot(data, patch_artist=True)

        ax.set_ylabel(statistic)

        for patch, colour in zip(box['boxes'], colours):
            patch.set_facecolor(colour)

        ax.set_xticklabels(names, rotation=90, fontsize=8)

        for k,v in kwargs.items():
            try:
                ax.set(**{k:v})
            except:
                pass
        ax.grid()
        return ax,box

    def plot_regression(self,site=None,freq='m',title="", size=20,**kwargs):
        '''
        Plot the model regression(s) for the specified site and frequency
        '''
        if site is None:
            site = list(self.obs.data.columns)
            stats_index = 'all'
        else:
            stats_index = site
            site = [site]

        tf = dt.validate_timeframe(freq).lower()

        ax = self._get_ax(kwargs)

        for m in self._iter_models(freq):
            _site_list = []
            for _site in site:
                if _site in m.data[tf].keys():
                    _site_list.append(_site)

            model_data = pd.DataFrame.from_dict(m.data[tf])[_site_list]
            obs_data = pd.DataFrame.from_dict(m.obs[tf])[_site_list]
            ax.scatter(obs_data,model_data,color=m.colour,s=size)

        ax.set_ylabel('model ' + self.var_name)
        ax.set_xlabel(str(self.ref_name))
        if isinstance(site, list):
            ax.set_title(title)
        else:
            ax.set_title(title + " %s" % site)

        ax.set(**kwargs)
        ax.grid()

        # plot regression lines and 1:1 line
        rl = get_ax_limit(ax)

        for m in self._iter_models(freq):
            try:
                mstats = m.stats[tf][stats_index]
            except KeyError:
                continue
            regress_line = mstats.loc['r_intercept'] + rl*mstats.loc['r_slope']
            ax.plot(rl,regress_line,color=m.colour,label=m.name)
        ax.plot(rl,rl,linestyle='--',color='black',label='1:1')

        ax.legend(loc='best')
        return ax

    def stat(self,statistic='mean',freq='m'):
        tf = dt.validate_timeframe(freq).lower()
        df = pd.DataFrame()

        for m in self._iter_models(freq):
            df[m.name] = m.stats[tf].loc[statistic]
        if statistic == 'mean':
            df[self.ref_name] = m.stats[tf].loc['obs_mean']

        return df

    def stat_percentiles(self,statistic='pearsons_r',freq='m',pctiles=None):
        '''
        Print a summary of percentiles for the specified statistic and timeframe
        '''
        if pctiles is None:
            pctiles = [0,5,25,50,75,95,100]
        tf = dt.validate_timeframe(freq).lower()
        df = pd.DataFrame()

        for m in self._iter_models(freq):
            if statistic == "grand_f":
                m_data = m.stats[tf].loc['fobj', m.stats[tf].columns != 'all']
                try:
                    stats = standard_percentiles(m_data)
                    df[m.name] = pd.Series(index=['grand_f'], data=[(stats['25%']+stats['50%']+stats['75%']+stats['100%'])/4])
                except IndexError:
                    logger.warning("no stats for model: %s",m.name)
            else:
                m_data = m.stats[tf].loc[statistic, m.stats[tf].columns != 'all']
                try:
                    df[m.name] = standard_percentiles(m_data, pctiles)
                except IndexError:
                    logger.warning("no stats for model: %s",m.name)

        return df.transpose()

    def data_percentiles(self,freq='m',pctiles=None):
        '''
        Print a summary of percentiles for the actual data values
        '''
        if pctiles is None:
            pctiles = [0,5,25,50,75,95,100]
        tf = dt.validate_timeframe(freq).lower()
        df = pd.DataFrame()

        if freq == 'd': # obs won't match model.obs since different obs.valid_idx for each model
            obs_series = self.obs.data.mean().values.flatten()
        else:
            pd_tf = dt.pandas_tf_dict[tf]
            obs_series = self.obs.data.resample(rule=pd_tf, how=self.aggr_how).mean().values.flatten()

        df[self.ref_name] = standard_percentiles(obs_series,pctiles)

        for m in self._iter_models(freq):
            m_data = pd.DataFrame.from_dict(m.data[tf]).mean().values.flatten()
            try:
                df[m.name] = standard_percentiles(m_data,pctiles)
            except IndexError:
                logger.warning("no stats for model: %s",m.name)

        return df.transpose()


def get_ax_limit(ax, space=np.linspace):
    ylim = ax.get_ylim()
    yscale = ax.get_yscale()
    xlim = ax.get_xlim()
    lim = [0.,1.]
    lim[0] = min(xlim[0],ylim[0])
    lim[1] = max(xlim[1],ylim[1])
    if yscale == 'log':
        if lim[0] < 0.:
            lim[0] = 0.1
        return np.logspace(np.log10(lim[0]),np.log10(lim[1]))
    else:
        return np.linspace(lim[0],lim[1])
