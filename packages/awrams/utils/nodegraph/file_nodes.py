import numpy as np
from .nodes import InputNode
from awrams.utils import mapping_types as mt
from awrams.utils.messaging.buffer_group import DataSpec
from awrams.utils.io.data_mapping import index_shape
from awrams.utils import geo, extents
from glob import glob

class SpatialFileNode(InputNode):
    def __init__(self,filename,variable=None,preload=False,force_grid=None):
        self.filename = filename
        self.variable = variable
        self.preload = preload

        #import h5py
        #self.fh = h5py.File(filename,'r')
        self.fh = _spatial_file_loader(filename,variable,force_grid)

        #lats,lons = self.fh['dimensions/latitude'][...], self.fh['dimensions/longitude'][...]
        self.cs = self.fh.get_coords()
        if preload:
            self.v = self.fh.data[...]
        else:
            self.v = self.fh.data
        self.dtype = self.v.dtype

        if len(self.cs) == 2:
            self._shaper = _2shaper
        elif len(self.cs) == 3:
            self._shaper = _3shaper
        else:
            raise Exception("Unsupported dimensions for spatial dataset")

    def close(self):
        self.fh.close()

    def get_data(self,coords):
        idx = self.cs.get_index(mt.CoordinateSet([coords.latitude,coords.longitude]))
        return self.v[idx].reshape(self._shaper(self.cs,coords))

    def get_dataspec(self):
        return DataSpec('array',[d.name for d in self.cs.dimensions],self.dtype)

    def get_coords(self):
        return self.cs

    def get_extent(self):
        return self.fh.get_extent()

def _2shaper(cs,coords):
    return coords.shape[1:]

def _3shaper(cs,coords):
    return tuple([cs.shape[0]] + list(coords.shape[1:]))

def _spatial_file_loader(fn,variable,force_grid=None):
    from os.path import splitext

    ext = splitext(fn)[1]

    if ext == '.flt':
        return GDALSpatialFile(fn,force_grid)
    elif ext in ['.nc','.h5']:
        return HDF5SpatialFile(fn,variable)
    else:
        raise Exception("Unsupported filetype", fn)

from awrams.utils.io.data_mapping import desimplify, index_shape

'''
Helpers from getting correct dimension info from HDF5
'''

from collections import OrderedDict

def get_dim_names(v):
    return [k.values()[0].name.split('/')[-1] for k in v.dims]
    
def get_dim_handles(v):
    return OrderedDict([(k.values()[0].name.split('/')[-1],k.values()[0]) for k in v.dims])

class NanFiller:
    def __init__(self,data,fill_value):
        self.data = data
        self.fill_value = fill_value
        self.dtype = data.dtype

    def __getitem__(self,idx):
        data = self.data[idx]
        data = data.reshape(index_shape(idx))
        data[data==self.fill_value] = np.nan
        return data   

class HDF5SpatialFile:
    def __init__(self,fn,variable):
        import h5py
        self.fh = h5py.File(fn,'r')
        self.dims = get_dim_names(self.fh[variable])#[k.label for k in self.fh[variable].dims]
        self._dim_handles = get_dim_handles(self.fh[variable])
        self.data = self.fh[variable]
        #self.data = NanFiller(self.fh[variable],self.fh[variable].fillvalue)

    def get_extent(self):
        # +++ Currently assumes no mask
        return extents.from_latlons(self._dim_handles['latitude'][...],self._dim_handles['longitude'][...])

    def get_coords(self):
        return mt.CoordinateSet([mt.Coordinates(mt.get_dimension(k,'',self._dim_handles[k].dtype),self._dim_handles[k][...]) for k in self.dims])

class GDALSpatialFile:
    def __init__(self,fn,force_grid=None):
        import gdal
        self.ds = gdal.Open(fn)
        self.data = GDALIndexWrapper(self.ds)
        self._grid_q = force_grid

    def get_extent(self):
        georef,mask = geo.get_georef_gridobj(self.ds,grid_q = self._grid_q)
        return extents.Extent(georef,mask=mask)
        
    def get_coords(self):
        georef,_ = geo.get_georef_gridobj(self.ds,False,grid_q = self._grid_q)
        return mt.latlon_to_coords(georef.latitudes.to_degrees(),georef.longitudes.to_degrees())

class GDALIndexWrapper:
    def __init__(self,ds):
        import gdal
        self.ds = ds
        rb = ds.GetRasterBand(1)
        self.fill_value = rb.GetNoDataValue()
        self.dtype = np.dtype(gdal.GetDataTypeName(rb.DataType).lower())
    
    def __getitem__(self,idx):
        if isinstance(idx,tuple):
            out_idx = []
            for c in idx:
                if isinstance(c,slice):
                    out_idx.append((c.start,c.stop-c.start))
                elif isinstance(c,int):
                    out_idx.append((c,1))
                else:
                    raise Exception('Unsupported index type', idx)
            data = self.ds.ReadAsArray(out_idx[1][0],out_idx[0][0],out_idx[1][1],out_idx[0][1])
        elif idx is ...:
            data = self.ds.ReadAsArray()
        else:
            raise Exception('Unsupported index type', idx)
        data[data==self.fill_value] = np.nan
        return data

class SpatialMultiFileNode(InputNode):
    def __init__(self,pattern,dimension,dim_func,variable=None,preload=False):

        files = glob(pattern)
        files.sort()

        if len(files) == 0:
            raise Exception("No files matching pattern", pattern)

        layer_dim = [dim_func(f) for f in files]

        self.variable = variable

        #import h5py
        #self.fh = h5py.File(filename,'r')

        self.file_layers = [_spatial_file_loader(filename,variable) for filename in files]

        self.ref_fh = self.file_layers[0]

        #lats,lons = self.fh['dimensions/latitude'][...], self.fh['dimensions/longitude'][...]
        self._flat_cs = self.ref_fh.get_coords()

        self.cs = mt.CoordinateSet((mt.Coordinates(dimension,layer_dim),self._flat_cs['latitude'],self._flat_cs['longitude']))

        if preload:
            self.v =[fh.data[...] for fh in self.file_layers]
        else:
            self.v = [fh.data for fh in self.file_layers]
        self.dtype = self.v[0].dtype

        self._shaper = _3shaper


    def close(self):
        self.fh.close()

    def get_data(self,coords):
        idx = self._flat_cs.get_index(mt.CoordinateSet([coords.latitude,coords.longitude]))

        out_shape = tuple([len(self.cs[0])] + list(index_shape(idx)))

        out_data = np.empty(out_shape,dtype=self.dtype)

        for i,v in enumerate(self.v):
            out_data[i] = v[idx]

        return out_data.reshape(self._shaper(self.cs,coords))

    def get_dataspec(self):
        return DataSpec('array',[d.name for d in self.cs.dimensions],self.dtype)

    def get_coords(self):
        return self.cs

    def get_extent(self):
        return self.ref_fh.get_extent()