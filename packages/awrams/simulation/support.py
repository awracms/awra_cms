import numpy as np
from awrams.utils import config_manager
import os
from awrams.utils.nodegraph import nodes

def build_output_mapping(model,outpath,save_vars=None,mode='a',dtype=np.float32,save_states_freq=None):
    '''
    Convenience function to save all models outputs to outpath as annual netCDF files
    '''

    ### populate output map with all model outputs
    io_settings = config_manager.get_system_profile().get_settings()['IO_SETTINGS']
    
    output_mapping = model.get_output_mapping()

    output_map_ncwrite = output_mapping.copy()
    
    if save_vars is None:
        save_vars = list(output_mapping)
    
    for k in save_vars:
        if k not in output_mapping:
            raise Exception("Variable %s not activated in model output settings" %k)

        output_map_ncwrite[k+'_ncsave'] = nodes.write_to_annual_ncfile(outpath,k,mode=mode,dtype=dtype)
        
    if save_states_freq is not None:
        state_keys = model.get_state_keys()
        state_path = os.path.join(outpath,'states')
        spatial_chunk = io_settings['CHUNKSIZES']['SPATIAL']
        for k in state_keys:
            if k not in output_mapping:
                raise Exception("State %s not activated in model output settings" %k)
            output_map_ncwrite[k+'_statesave'] = nodes.write_to_ncfile_snapshot(state_path,k,mode=mode, \
                                                                                freq=save_states_freq,dtype=np.float64, \
                                                                                chunksizes = (1,spatial_chunk,spatial_chunk))

    return output_map_ncwrite
