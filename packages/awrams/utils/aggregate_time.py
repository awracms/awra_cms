import os

import awrams.utils.datetools as dt
from awrams.utils.io.general import h5py_cleanup_nc_mess
from awrams.utils.datetools import resample_dti,truncate_resample_dti,truncate_dti
from awrams.utils.processing.time_conversion import resample_data

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('daily_monthly_sched')

def process(var_map,out_path,period,to_freq,method='mean',file_mode='w'):
    '''

    :param var_map: {var_name:file_name(with wildcard)
    :param period: pandas DatetimeIndex
    :param to_freq: monthly or annual
    :param method: mean or sum
    :param file_mode: w to replace existing or a to append
    '''

    if to_freq.lower() == 'monthly':
        to_freq = 'M'
    elif to_freq.lower() == 'annual':
        to_freq = 'A'

    for variable in var_map:
        h5py_cleanup_nc_mess()
        try:
            logger.info("Converting to %s: %s", dt.tf_dict[to_freq.upper()].lower(),variable)
        except KeyError:
            logger.info("Converting to %s: %s", to_freq,variable)

        in_path,in_pattern = os.path.split(var_map[variable])

        period = truncate_dti(period,to_freq)

        if len(period) > 0:
            method = 'mean' if method == 'mean' else 'sum'
            logger.info("Using method %s", method)

            resample_data(in_path,in_pattern,variable,period,out_path,to_freq,method,mode=file_mode,enforce_mask=True,extent=None)

    logger.info("Temporal aggregation completed")

if __name__ == '__main__':
    pass
    # var_map = {'s0_avg':'/data/cwd_awra_data/awra_test_outputs/Scheduled_v5_sdcvd-awrap01/s0_avg_*.nc'}
    # period = dt.dates('15 may 2013','20 feb 2015')
    # opath = '/data/cwd_awra_data/awra_test_outputs/_test/4'
    # process(var_map,opath,period,'M')

    # var_map = {'s0_avg':'/data/cwd_awra_data/awra_test_outputs/Scheduled_v5_sdcvd-awrap01/s0_avg_*.nc'}
    # period = dt.dates('15 may 2013','20 feb 2015')
    # opath = '/data/cwd_awra_data/awra_test_outputs/_test/6'
    # process(var_map,opath,period,'annual')

    # var_map = {'s0_avg':'/data/cwd_awra_data/awra_test_outputs/Scheduled_v5_sdcvd-awrap01/processed/values/month/s0_avg.nc'}
    # period = dt.dates('15 nov 2013','20 feb 2015')
    # opath = '/data/cwd_awra_data/awra_test_outputs/_test/5'
    # process(var_map,opath,period,'A')

    # period = dt.dates('15 jun 2013','20 feb 2015')
    # opath = '/data/cwd_awra_data/awra_test_outputs/_test/7'
    # process(var_map,opath,period,'A-JUN')
    # var_map = {'s0_avg':'/data/cwd_awra_data/awra_test_outputs/_test/4/s0_avg.nc'}
    # period = dt.dates('15 may 2013','20 feb 2015')
    # opath = '/data/cwd_awra_data/awra_test_outputs/_test/7'
    # process(var_map,opath,period,'A-JUN')
