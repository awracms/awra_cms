'''
benchmarking setup for various comparisons
'''
from string import Template
import numpy as np
import pandas as pd

from .utils import infer_freq,extract_sites
from awrams.utils.metatypes import ObjectDict

from awrams.utils import config_manager

system_profile = config_manager.get_system_profile().get_settings()

SM_MODEL_VARNAMES = system_profile['BENCHMARKING']['SM_MODEL_VARNAMES']
SM_MODEL_LAYERS = system_profile['BENCHMARKING']['SM_MODEL_LAYERS']

class Selector(ObjectDict):
    def __init__(self, names):
        self._sel = []
        self._build(names)

    def _build(self, names):
        from functools import partial

        def select(n):
            if not n in self._sel:
                self._sel.append(n)
        def unselect(n):
            if n in self._sel:
                self._sel.remove(n)

        for name in names:
            self._sel.append(name)
            self.__dict__[name] = ObjectDict()
            self.__dict__[name]['select'] = partial(select, name)
            self.__dict__[name]['unselect'] = partial(unselect, name)

    def _add(self, k):
        if not k in self:
            self._build([k])

    def __repr__(self):
        return repr(self._sel)

    def __call__(self):
        return self._sel


class Model(object):
    def __init__(self, name):
        self.name = name

    def infer_freq(self, df):
        self.freq = infer_freq(df)

    def load_from_csv(self, csv):
        df = pd.read_csv(csv,index_col=0,parse_dates=True)
        self.infer_freq(df)
        return df


class SoilMoistureModel(Model):
    def __init__(self, site_info):
        self.site_ref = "Unique_ID"
        self.site_info = site_info
        self.model_thickness = []
        model_layers = []
        for v in SM_MODEL_VARNAMES:
            self.model_thickness.append(SM_MODEL_LAYERS[v])
            model_layers.append(sum(self.model_thickness))
        self.model_layers = np.array(model_layers)

    def derive_soil_moisture(self, data_path, layer_names, period):
        """derive volumetric soil moisture for each observed layer"""
        sites = list(self.site_info.keys())
        site_idx = [self.site_info[site][self.site_ref] for site in sites]
        self.get_data_at_sites(data_path, sites, site_idx, period)

        ddf = dict(list(zip(layer_names,[dict() for l in layer_names])))

        for layer_name in layer_names:
            ddf[layer_name] = pd.DataFrame(columns=[sites], index=period)
            for site in sites:
                obs_layer = self.site_info[site]['layer'][layer_name]
                overlap = [0.]*len(self.model_thickness)

                top = np.searchsorted(self.model_layers,obs_layer[0])
                bottom = np.searchsorted(self.model_layers,obs_layer[1])

                if top == bottom:  # obs layer contained in 1 model layer
                    overlap[top] = (obs_layer[1] - obs_layer[0]) / self.model_thickness[top]
                else:
                    overlap[top] = (self.model_layers[top] - obs_layer[0]) / self.model_thickness[top]
                    if bottom - top > 1:
                        for l in range(top+1,bottom-1):
                            overlap[l] = 1.
                    overlap[bottom] = (obs_layer[1] - self.model_layers[bottom-1]) / self.model_thickness[bottom]

                ddf[layer_name][site] = np.array([self.data[i][site] * overlap[i] for i,thk in enumerate(self.model_thickness)]).sum(axis=0)

                ddf[layer_name][site] /= (obs_layer[1] - obs_layer[0])         # convert to fraction full
                ddf[layer_name][site] *= 100.                                  # convert to % volSM
                ddf[layer_name][site] += self.site_info[site]["wp_A_horizon"]  # add wilting point fullness
        return ddf

    def get_data_at_sites(self, data_path, sites, site_idx, period):
        self.data = {}
        for i,v in enumerate(SM_MODEL_VARNAMES):
            csv = Template(data_path).substitute(v=v)
            df = self.load_from_csv(csv)
            self.data[i] = extract_sites(df, sites, site_idx, period)

    def wilting_point(self):
        """either 1. or 0. ie do or don't apply"""
        return 1.


