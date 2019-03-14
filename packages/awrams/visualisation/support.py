import numpy as np
import pandas as pd

import awrams.utils.datetools as dt
from awrams.utils.metatypes import ObjectDict,ObjectContainer,ObjectList
from awrams.utils.ts.processing import mean_over_time,sum_over_time,spatial_aggregate


def _sanitize_inputs(inputs):
    if isinstance(inputs, VariableInstance):
        return VariableGroup(inputs.source, [inputs])
    if isinstance(inputs, VariableGroup):
        return inputs
    return VariableGroup(inputs[0].source, inputs)

def _sanitize_aggregate_method(variable):
    try:
        agg_method = variable.agg_method
    except KeyError:
        print("No aggregation method property for %s, defaulting to mean" % variable.name)
        agg_method = 'mean'

    if agg_method == 'mean':
        return mean_over_time
    else:
        return sum_over_time

def _sanitize_period(period, freq):
    if type(period) == str or type(period) == int:
        period = dt.dates(period)
        return pd.date_range(period[0],period[-1],freq=freq)
    ### +++ can it handle "jan 2000 - mar 2001" OR a pd.DatetimeIndex???
    elif type(period) == pd.DatetimeIndex:
        return pd.date_range(period[0],period[-1],freq=freq)
    else:
        raise Exception('period no good')


class VariableInstance(object):
    def __init__(self,source,name,units):
        self.source = source # AWRAResults object
        self.name = name
        self.units = units
        self.meta = ObjectDict()

    def get_data(self,period,extent): #pylint: disable=unused-argument
        raise Exception("Not implemented")

    def __repr__(self):
        return self.name


class VariableGroup(ObjectContainer):
    '''
    Iterable container for variables, with metadata
    '''

    def __init__(self,source,variables=None):
        if variables is None:
            variables = []

        self.source = source
        ObjectContainer.__init__(self)
        for v in variables:
            if v.name in self: ### ie same variable from different results objects
                if v.source.name is None:
                    raise Exception("trying to add variable that already exists, need to give results object a name")
                was = self[v.name]
                self._remove_key(v.name)
                self[was.name + "_" + was.source.name] = was
                self[v.name + "_" + v.source.name] = v
            else:
                self[v.name] = v


class DataVariable(VariableInstance):
    '''
    Wrapper for creation of AWRA variables around existing data
    '''
    def __init__(self,source,name,units,data):
        #+++ Needs to know own extent?
        VariableInstance.__init__(self,source,name,units)
        self._data = data

    def get_data(self,period,extent):
        new_extent = extent.translate_to_origin(global_georef())
        return np.ma.MaskedArray(data=self._data[new_extent.indices],mask=new_extent.mask)

