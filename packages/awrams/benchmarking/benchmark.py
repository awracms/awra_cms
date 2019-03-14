'''
benchmarking setup for various comparisons
'''
import pandas as pd
import numpy as np
import os

import awrams.utils.datetools as dt
from awrams.utils.metatypes import ObjectDict

from .comparison import ComparisonSet
from .utils import get_catchments_by_ids, get_sites_by_ids
from .model import SoilMoistureModel,Model
from awrams.utils import config_manager

system_profile = config_manager.get_system_profile().get_settings()

SM_OBSERVED_LAYERS = system_profile['BENCHMARKING']['SM_OBSERVED_LAYERS']


class Benchmark:
    """
    facilitates comparison of observed against model derived data
    """
    def __init__(self, obs_name, var_name):
        """
        :param obs_name: observations label
        :param var_name: model data label
        :return:
        """
        self.obs_name = obs_name
        self.var_name = var_name
        self.annual_rule = 'A'
        self.period = None

        self.site_ref_type = 'ID'  # see column names in SiteLocationsWithIndex.csv
        self.site_id_type = 'Unique_ID'   # see column names in SiteLocationsWithIndex.csv
        self.id_type = 'catchment'

        self.site_set_name = None
        self.how = np.sum

        self.benchmark = None

    def _load_data(self, data_csv, id_list=None, convert_units=1.0):
        df = self._load_from_csv(data_csv)
        df = self._load_extents(df, id_list)
        df *= convert_units
        if df.empty: raise Exception('dataframe is empty...if id_list supplied does it intersect dataframe columns?')
        return df

    def load(self, data_csv, id_list=None, convert_units=1.0):
        """
        load observed data from csv

        :param data_csv: csv containing observations (see example datasets in data folder)
        :param id_list: None for comparison of all ids in csv or list of ids for subset
        :param convert_units: factor to apply to observations
        :return:
        """
        df = self._load_data(data_csv, id_list, convert_units)

        self.sites = list(self._extents.keys())

        self.benchmark = ComparisonSet(df, self.obs_name, self.var_name, self.period, how=self.how, annual_rule=self.annual_rule)

    def add_model(self, name, data_csv, convert_units=1.0):
        """
        add a model dataset for comparison

        :param name: model label
        :param data_csv: csv containing model data
        :param convert_units: factor to apply to model data
        :return:
        """
        model = Model(name)
        model.df = model.load_from_csv(data_csv)

        # do we need to convert column names from 'pred_index' to 'ID'
        if not any([site in model.df.columns for site in self.sites]):
            for site in self.sites:
                model.df.rename(columns={self._site_map[site]: site}, inplace=True)
        else:
            try:
                model.df = model.df[self.sites]
            except KeyError:
                pass

        self.benchmark._add_model(model.df * convert_units,name,freq=model.freq)

    def _load_from_csv(self, csv):
        df = pd.io.parsers.read_csv(csv, index_col=0, parse_dates=True, dayfirst=True, na_values=['NaN','NA'])

        if self.period is None:
            self.period = dt.dates(df.index[0], df.index[-1])
        else:
            df = df.loc[(df.index >= self.period[0])&(df.index <= self.period[-1])]

        return df

    def _load_extents(self, df, id_list):
        # get site/catchment ids from obs df columns
        if id_list is None:
            if self.id_type == 'catchment':
                # self._extents = get_catchments_by_ids(df.columns)
                self._extents = {cat:None for cat in df.columns}
                _df = df
            elif self.id_type == 'site':
                _df = pd.DataFrame()
                self._extents, self._site_map = get_sites_by_ids(df.columns, self.site_id_type, self.site_ref_type, self.site_set_name)
                for site_name,site_id in list(self._site_map.items()):
                    try:
                        _df[site_name] = df[site_id]
                    except KeyError:
                        _df[site_name] = df[site_id.lower()]

        else: # site/catchment ids explicitly supplied in csv
            if self.id_type == 'catchment':
                # self._extents = get_catchments_by_ids(id_list)
                # _df = pd.DataFrame()
                # for cat in list(self._extents.keys()):
                #     _df[cat] = df[cat]
                self._extents = {}
                _df = pd.DataFrame()
                for cat in id_list:
                    self._extents[cat] = None
                    _df[cat] = df[cat]

            elif self.id_type == 'site':
                _df = pd.DataFrame()
                self._extents, self._site_map = get_sites_by_ids(id_list, self.site_id_type, self.site_ref_type, self.site_set_name)
                for site_name,site_id in list(self._site_map.items()):
                    try:
                        _df[site_name] = df[site_id]
                    except KeyError:
                        _df[site_name] = df[site_id.lower()]
        return _df

class BenchmarkSoilMoisture(Benchmark):
    """
    facilitates comparison of observed against model derived soil moisture data
    """
    def __init__(self, obs_name, var_name, sites_meta):
        """
        :param obs_name: observations label
        :param var_name: model data label
        :param sites_meta: meta data describing observation sites
        :return:
        """
        super(BenchmarkSoilMoisture,self).__init__(obs_name, var_name)

        self.sites_meta = sites_meta
        self.layers = SM_OBSERVED_LAYERS

        self.site_ref_type = 'ID'  # see column names in SiteLocationsWithIndex.csv
        self.site_id_type = 'Unique_ID'   # see column names in SiteLocationsWithIndex.csv
        self.id_type = 'site'

        self.how = np.mean

    def load(self, csv_path, id_list=None, convert_units=1.0):
        """
        load observed data from csv

        :param csv_path: path to csvs containing observations
                         expects to find csv files: sm_top.csv,sm_shallow.csv,sm_middle.csv,sm_deep.csv,sm_profile.csv
        :param id_list: None for comparison of all ids in csv or list of ids for subset
        :param convert_units: factor to apply to observations
        :return:
        """
        self.benchmark = ObjectDict()

        for layer in self.layers:
            df = self._load_data(os.path.join(csv_path, layer+'.csv'), id_list, convert_units)

            self.benchmark[layer] = ComparisonSet(df,self.obs_name,self.var_name,self.period,how=self.how,annual_rule=self.annual_rule)

        self.sites = list(self._extents.keys())
        self.cfg = {s: self.sites_meta[s] for s in self.sites}

        self.add_model = self._add_model

    def _add_model(self, name, csv_path, convert_units=1.0):
        model = SoilMoistureModel(self.cfg)
        ddf = model.derive_soil_moisture(csv_path,self.layers,self.period)

        for layer in self.layers:
            self.benchmark[layer]._add_model(ddf[layer],name,freq=model.freq)
