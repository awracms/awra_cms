'''
General grab bag of helper functions
'''

import numpy as np
import os

#from .settings import DEFAULT_AWRAL_MASK
from .precision import iround, quantize, aquantize, sanitize_cell, sanitize_geo_array

def _DEP_load_mask(fn_mask):
    '''
    Identify the AWAP cells required for the continental AWRA-L run.
    Returns a list of 2 element tuples, with each element the indices of a AWAP cell.
    '''
    return list(zip(*np.where(np.logical_not(load_mask_grid(fn_mask)))))

def _DEP_load_mask_grid(fn_mask):
    if os.path.splitext(fn_mask)[1] == '.flt':
        return _load_mask_flt(fn_mask)
    elif os.path.splitext(fn_mask)[1] == '.h5':
        return _load_mask_h5(fn_mask)
    else:
        raise Exception("unknown mask grid format: %s" % fn_mask)

def _DEP_load_mask_h5(fn_mask):
    import h5py
    h = h5py.File(fn_mask,'r')
    return h['parameters']['mask'][:] <= 0

def _DEP_load_mask_flt(fn_mask):
    import osgeo.gdal as gd
    gd_mask = gd.Open(fn_mask)
    bd_mask = gd_mask.GetRasterBand(1)
    return bd_mask.ReadAsArray() <= 0

def _DEP_load_meta():
    import pandas as _pd
    import os as _os
    # from settings import AWRAPATH as _AWRAPATH

    #TODO - metadata csv should it be a module

    p = _os.path.join(_os.path.dirname(__file__),'data','awral_outputs.csv')
    output_meta = _pd.DataFrame.from_csv(p)

    # Read input metadata into dataframe as well and concat it with output metadata
    # input_meta = _pd.DataFrame.from_csv(_os.path.join(_AWRAPATH,"Landscape/Metadata/awraL_inputs.csv"))

    # return _pd.concat([output_meta, input_meta])
    return output_meta

def print_error(message):
    import sys
    print(message,file=sys.stderr)

class IndexGetter:
    '''
    Helper class for using index creation shorthand
    eg IndexGetter[10:,5] returns [slice(10,None),5]
    '''
    def __getitem__(self,indices):
        return indices

class Indexer:
    '''
    Wrapper class that refers it's get/set item methods to another function
    '''
    def __init__(self,getter_fn,setter_fn = None):
        self.getter_fn = getter_fn
        self.setter_fn = setter_fn

    def __getitem__(self,idx):
        return self.getter_fn(idx)

    def __setitem__(self,idx,value):
        return self.setter_fn(idx,value)

index = IndexGetter()

def as_int(n):
    return int(np.floor(n))

def shuffle(source,indices):
    return [source[i] for i in indices]