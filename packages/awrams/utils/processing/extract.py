from awrams.utils.io import data_mapping as dm
import pandas as pd
import numpy as np

def _get_max_chunk():
    from awrams.utils import config_manager

    sys_settings = config_manager.get_system_profile().get_settings()

    return sys_settings['IO_SETTINGS']['MAX_EXTRACT_CHUNK']

def extract(path,pattern,var_name,extent_map,period):
    '''
    Extract spatial aggregations (mean by area)
    '''

    sfm = dm.SplitFileManager.open_existing(path,pattern,var_name)
    
    return extract_from_filemanager(sfm,extent_map,period)

def extract_from_filemanager(sfm,extent_map,period):
    '''
    Extract spatial aggregations (mean by area) for an existing SplitFileManager
    '''

    
    # Calculate chunksizes
    MAX_EXTRACT_CHUNK = _get_max_chunk()
    
    max_spatial = max([np.prod(v.shape) for v in extent_map.values()])
    dsize = sfm.mapped_var.dtype.itemsize
    
    max_plen = int(np.floor(MAX_EXTRACT_CHUNK / (max_spatial*dsize)))
    
    chunked_p = sfm.get_chunked_periods(max_plen)
    split_p = [p for p in [period.intersection(cp) for cp in chunked_p] if len(p) > 0]
    
    df = pd.DataFrame(index=period,columns=extent_map.keys())
    
    eweights = {}
    for k,extent in extent_map.items():
        eweights[k] = extent.areas/extent.area
    
    for p in split_p:
        for k,extent in extent_map.items():
            data = sfm.get_data(p,extent)
            dataw = data * eweights[k]
            df[k].loc[p] = dataw.sum(axis=(1,2))
    
    return df

def extract_gridded_stats(sfm,period,extent):
    '''
    Extract min/max/mean over the specified period and extent
    '''

    
    # Calculate chunksizes
    MAX_EXTRACT_CHUNK = _get_max_chunk()
    
    max_spatial = np.prod(extent.shape)
    dsize = sfm.mapped_var.dtype.itemsize
    
    max_plen = int(np.floor(MAX_EXTRACT_CHUNK / (max_spatial*dsize)))
    
    chunked_p = sfm.get_chunked_periods(max_plen)
    split_p = [p for p in [period.intersection(cp) for cp in chunked_p] if len(p) > 0]
    
    for p in split_p:
        for k,extent in extent_map.items():
            data = sfm.get_data(p,extent)
            dataw = data * eweights[k]
            df[k].loc[p] = dataw.sum(axis=(1,2))
    
    return df


