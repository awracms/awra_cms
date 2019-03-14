import re
import sys
import numpy as np
import datetime as dt

from awrams.utils.metatypes import ObjectDict
from awrams.utils.helpers import iround

# +++ Should probably stop logging from this file...
from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('netcdf_wrapper')

#from awrams.utils.settings import DEFAULT_CHUNKSIZE, VAR_CHUNK_CACHE_SIZE, VAR_CHUNK_CACHE_NELEMS, VAR_CHUNK_CACHE_PREEMPTION, VARIABLE_PRECISION, DEFAULT_PRECISION #pylint: disable=no-name-in-module

def set_chunk_cache(dataset, variable, **params):
    p = dict(var_chunk_cache_size=VAR_CHUNK_CACHE_SIZE,
             var_chunk_cache_nelems=VAR_CHUNK_CACHE_NELEMS,
             var_chunk_cache_preemption=VAR_CHUNK_CACHE_PREEMPTION)
    p.update(**params)

    dataset.variables[variable].set_var_chunk_cache(size=p['var_chunk_cache_size'],
                                                    nelems=p['var_chunk_cache_nelems'],
                                                    preemption=p['var_chunk_cache_preemption'])

def extract_from(fn,path):
    """
    Returns the entire contents from a variable at a given _path_ within
    a netcdf file. Path should be / separated.

    Returns None if there is no data at the given path
    """
    import netCDF4 as nc

    dataset = nc.Dataset(fn,'r')
    try:
        pathComponents = [p for p in path.split('/') if len(p)]
        groupNames = pathComponents[0:-1]
        target = pathComponents[-1]
        data = dataset
        for group in groupNames:
            if not group in data.groups:
                return None

            data = data.groups[group]

        if target in data.variables:
            return data.variables[target][...]
        elif target in data.groups:
            return extract_all(data.groups[target])
        return None
    finally:
        dataset.close()

def extract_all(data):
    result = {}
    for grp in list(data.groups.keys()):
        result[grp] = extract_all(data.groups[grp])

    for var in list(data.variables.keys()):
        result[var] = data.variables[var][...]

    for attr in data.ncattrs():
        result[attr] = data.getncattr(attr)

    return result

def epoch_from_nc(ncfile):
    """
    Extract the epoch from a netCDF data set and return as a datetime.

    (Assumes the netCDF dataset schema matches those used in AWRAMSI -
    notably that there is a 'time' variable with the epoch expressed
    in the units property)
    """
    epoch = ncfile.variables['time'].units.split(' ')[2]
    return dt.datetime.strptime(epoch,'%Y-%m-%d')

def dataset_frequency(dataset):
    """
    Return the date of the first time step in a dataset
    """
    import pandas as pd
    epoch = epoch_from_nc(dataset).toordinal()
    time_dim = dataset.variables['time'][:]
    diff = np.diff(time_dim)
    if diff.max() == 1:
        return 'daily'
    elif diff.min() >= 28 and diff.max() <= 31:
        return 'monthly'
    elif diff.min() >= 365 and diff.max() <= 366:
        return 'yearly'
    else:
        raise Exception("unexpected frequency from time index")

def start_date(dataset):
    """
    Return the date of the first time step in a dataset
    """
    epoch = epoch_from_nc(dataset)
    return dt.datetime.fromordinal(epoch.toordinal()+
                                   int(dataset.variables['time'][0]))

def end_date(dataset,day_exist_chn_name=None):
    """
    Return the date of the last time step in a dataset,
    paying attention to a 'day exists?' variable if instructed
    """
    epoch = epoch_from_nc(dataset)
    if day_exist_chn_name is not None: # exist channel in climate inputs only
        day_exist = dataset.groups['provenance'].variables[day_exist_chn_name][:]
        #if type(day_exist) == np.ma.core.MaskedArray:
        if hasattr(day_exist,'mask'):
            return dt.datetime.fromordinal(epoch.toordinal() + \
                    int(dataset.variables['time'][day_exist.compressed()[-1]]))
        else:
            # return  dt.datetime.fromordinal(epoch.toordinal()+int(dataset.variables['time'][-1]))
            return  dt.datetime.fromordinal(epoch.toordinal()+int(dataset.variables['time'][day_exist[-1]]))

    else:
        try:
            return  dt.datetime.fromordinal(epoch.toordinal()+int(dataset.variables['time'][-1]))
        except np.ma.core.MaskError:
            return dt.datetime.fromordinal(epoch.toordinal()+int(dataset.variables['time'][:].compressed()[-1]))

def write_nested_data(dataset,group,data,warn_on_different_attributes=True,change_resolvers=None):
    if change_resolvers is None:
        change_resolvers = {}
    grp = dataset
    data = {group:data}
    _write_data_to_group(grp,data,warn_on_different_attributes,change_resolvers)

def _write_data_to_group(group,data,warn_on_different_attributes,change_resolvers):
    for key,value in list(data.items()):
        if hasattr(value,'keys'):
            #print group
            sub_grp = _create_or_use_group(group,key)
            _write_data_to_group(sub_grp,value,warn_on_different_attributes,change_resolvers)
        elif hasattr(value,"apply_to"):
            sub_grp = _create_or_use_group(group,key)
            value.apply_to(sub_grp)
        else:
            try:
                if warn_on_different_attributes and (key in group.ncattrs()):
                    old_value = group.getncattr(key)

                    #numpy and python floats have different str representations; check for equivalence
                    if isinstance(old_value,float):
                        changed = old_value != value
                    else:
                        changed = str(old_value) != str(value)
                    if changed:# or (np.shape(old_value) != np.shape(value)):
                        if key in change_resolvers:
                            value_to_use = change_resolvers[key](old_value,value)
                            group.setncattr(key,value_to_use)
                            continue
                        else:
                            logger.warn("Changing value of %s attribute in %s (old value=%s,new value = %s)",key,group.path,str(old_value),str(value))
                group.setncattr(key,value)
            except TypeError as e:
                msg = "Error writing to dataset attribute %s, with value %s. Error %s"%(key,value,str(e))
                raise Exception(msg).with_traceback(sys.exc_info()[2])

def _create_or_use_group(parent_group,new_group):
    if new_group in parent_group.groups:
        return parent_group.groups[new_group]

    return parent_group.createGroup(new_group)

def _create_or_use_dimension(dest,dim,size):
    if dim in dest.dimensions:
        return dest.dimensions[dim]

    return dest.createDimension(dim,size)

def _create_or_use_variable(dest,var,dtype,dims):
    if var in dest.variables:
        return dest.variables[var]

    return dest.createVariable(var,dtype,dims)


def dimensions_from_georef(gr):
    idim = NCDimension(dtype=np.dtype('int32'),
                       size=None,
                       data=None,
                       meta=ObjectDict(units="days since 1900-01-01",
                                       calendar = "gregorian",
                                       name = "time",
                                       long_name = "time",
                                       standard_name='time'))

    ydim = NCDimension(size=gr['nlats'],
                       dtype=np.dtype('float64'),
                       meta=ObjectDict(standard_name='latitude',
                                       long_name='latitude',
                                       name='latitude',
                                       units='degrees_north'),
                       data=gr['lat_origin'] - gr['cellsize'] * np.arange(gr['nlats'], dtype=np.float64))

    xdim = NCDimension(size=gr['nlons'],
                       dtype=np.dtype('float64'),
                       meta=ObjectDict(standard_name='longitude',
                                       long_name='longitude',
                                       name='longitude',
                                       units='degrees_east'),
                       data=gr['lon_origin'] + gr['cellsize'] * np.arange(gr['nlons'], dtype=np.float64))

    return (idim, ydim, xdim)

class NCDimension(ObjectDict):
    def __init__(self, **pars):
        self.nc_par = ObjectDict(zlib=False) #fill_value=-999., zlib=False)
        self.update(pars)

class NCVariableParameters(ObjectDict):
    def __init__(self, **pars):
        self.fill_value = -999.
        self.zlib = True #False #True
        self.complevel = 1 #None #1 #9
        self.shuffle = True #False
        self.chunksizes = None
        self.update(pars)

class NCVariable(ObjectDict):
    def __init__(self, **pars):
        self.nc_par = NCVariableParameters()
        self.update(pars)

def create_dimension(ncd, par):
    ncd.createDimension(par.meta.name, par.size)
    create_variable(ncd, par, (par.meta.name,))
    if par.data is not None:
        ncd.variables[par.meta.name][:] = par.data
    ncd.sync()

def create_variable(ncd, par, dims):
    #+++ NetCDF bug workaround
    # (Some versions of) NetCDF will silently fail on creating a variable
    # if it's dimensions are less than the specified chunksize
    if 'chunksizes' in par.nc_par:
        if par.nc_par.chunksizes:
            chunksizes = []
            for i, d in enumerate(ncd.dimensions):
                if ncd.dimensions[d].isunlimited():
                    chunksizes.append(par.nc_par.chunksizes[i])
                else:
                    dim_sz = len(ncd.dimensions[d])
                    if dim_sz == 0: dim_sz = 1
                    chunksizes.append(min(dim_sz,par.nc_par.chunksizes[i]))
            par.nc_par.chunksizes = chunksizes

    v = ncd.createVariable(par.meta.name, par.dtype, dimensions=dims, **par.nc_par)
    if 'meta' in par:
        v.setncatts(dict(par.meta))
    ncd.sync()

def add_group(nco, pars):
    if 'cmp_dtype' in list(pars.keys()):
        ###comp_type = np.dtype([('Orbit', 'i'), ('Location', np.str_, 6), ('Temperature (F)', 'f8'), ('Pressure (inHg)', 'f8')])
        nco.create_dataset('climate', shape=(pars['size'],), dtype=pars['cmp_dtype'])
        #nco.createCompoundType(pars['cmp_dtype'], 'climate')
        #return nco
    else:
        for k, v in pars.items():
            if hasattr(v, 'keys'): ### nested parameters
                if not k in nco.groups:
                    ncc = nco.createGroup(k)
                else:
                    ncc = nco.groups[k]
                add_group(ncc, v)
        #return ncc

def add_group_attrs(nco, pars):
    for k, v in pars.items():
        if hasattr(v, 'keys'): ### nested parameters
            if not k in nco.groups:
                ncc = nco.createGroup(k)
            else:
                ncc = nco.groups[k]
            add_group_attrs(ncc, v)
        else:
            nco.attrs.create(k, v, shape=(1,))

def create_awra_ncd(nc_file, variables, dims):

    import netCDF4 as nc
    
    ncd = nc.Dataset(nc_file, 'w', 'NETCDF4')

    for dim in dims:
        create_dimension(ncd, dim)

    for v in variables:
        create_variable(ncd, v, v.dims)
    ncd.setncatts(dict(var_name=variables[0].meta.name))
    ncd.setncatts(dict(i_dim=dims[0].meta.name))
    ncd.setncatts(dict(x_dim=dims[1].meta.name))
    ncd.setncatts(dict(y_dim=dims[2].meta.name))
    #ncd.close()
    return ncd

def create_provenance_group(ncd, prov_obj, dim_name): #, with_exist=False):
    ncg = ncd.createGroup('provenance')
    # cmp_dtype=np.dtype([('index_value', 'i'),
    #                     ('number_of_stations_reporting', 'i'),
    #                     ('last_updated_epoch', 'l'),
    #                     ('last_updated', 'S1', 32),
    #                     ('hashlib_md5', 'S1', 32),
    #                     ('filename', 'S1', 256),
    #                     ('analysis_version', 'S1', 32),
    #                     ('analysis_time', 'S1', 32)])
    # nc_cmp_dt = ncg.createCompoundType(prov_obj.cmp_dtype, 'provenance_dtype')
    # ncc = ncg.createVariable('source', nc_cmp_dt, (dim_name,))
    
    ncc = {}
    for v in prov_obj.meta.keys():
        ncc[v] = ncg.createVariable(v, prov_obj.meta[v]['dtype'], (dim_name,))
        if ncd.variables[dim_name].shape[0] > 0:
            if type(prov_obj.meta[v]['null']) == str:
                ncc[v][:] = np.array([prov_obj.meta[v]['null']]*ncd.variables[dim_name].shape[0],dtype=prov_obj.meta[v]['dtype'])
    return ncc

def get_provenance(ncd):
    ncg = ncd.groups['provenance']
    ncp = {}
    for v in ncg.variables.keys():
        ncp[v] = ncg.variables[v]
    return ncp
    
def save_provenance(ncd, prov_dict, i):
    ncg = ncd.groups['provenance']
    for v in prov_dict.keys():
        ncv = ncg.variables[v]
        ncv[i] = prov_dict[v]
    ncg.variables['exist'][i] = i
