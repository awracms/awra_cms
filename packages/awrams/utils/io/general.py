from awrams.utils.awrams_log import get_module_logger
from awrams.utils.config_manager import get_system_profile
logger = get_module_logger('awrams.utils.io')

def open_append(db_opener,fn,mode='a'):
    #import netCDF4 as nc
    import time
    ### +++ Review
    ### sometimes crashes with HDF error, try 5 times then fail
    not_opened = 0
    while not_opened < 5:
        try:
            ds = db_opener(fn,mode)
            #ds = nc.Dataset(fn,mode)
            not_opened = 5
        except OSError: #RuntimeError:
            logger.warning("trouble opening (attempt %d/5): %s", (not_opened + 1), fn)

            h5py_cleanup_nc_mess(fn)

            not_opened += 1
            if not_opened == 5:
                raise
            time.sleep(5)
    return ds

# +++ db_open_with still appears to get used in a few places
def db_open_with(opener=None):
    from awrams.utils.io import db_helper
    #import awrams.utils.settings as settings

    io_settings = get_system_profile().get_settings()['IO_SETTINGS']

    if opener is None:
        opener = io_settings['DB_OPEN_WITH']

    try:
        opener_method = getattr(db_helper,opener)
    except:
        import netCDF4
        opener_method = netCDF4.Dataset

    return opener_method

    if opener == '_h5py':
        return db_helper._h5py
    elif opener == '_nc':
        return db_helper._nc
    else:
        import netCDF4
        return netCDF4.Dataset

def h5py_cleanup_nc_mess(fn=None,show_log=False):
    import h5py
    import os
    ids = h5py.h5f.get_obj_ids(h5py.h5f.OBJ_ALL,h5py.h5f.OBJ_FILE)
    for id in ids:
        if fn is not None:
            if os.path.normpath(id.name.decode("utf-8")) == os.path.normpath(fn):
                if show_log:
                    logger.warning("lingering file id detected....closing")
                try:
                    id.close()
                except:
                    pass
        else:
            try:
                if show_log:
                    logger.warning("lingering file id detected....closing")
                id.close()
            except:
                pass
