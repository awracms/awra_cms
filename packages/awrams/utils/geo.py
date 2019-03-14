from numbers import Number
import numpy as np
from enum import Enum

class GeoUnit:
    def __init__(self,value):
        self.value = np.round(value)
            
    @classmethod
    def from_dms(self,d=0,m=0,s=0):
        return GeoUnit(1e6 * (d*3600.0 + m * 60.0 + s))
    
    def to_degrees(self):
        return self.value / (3600.0*1e6)

    def quantize(self,qunit):
        if not isinstance(qunit,GeoUnit):
            qunit = GeoUnit.from_dms(qunit)
        d = self/qunit
        d = np.round(d)
        return qunit * d
        
    def __repr__(self):
        return("%s" % self.to_degrees())
    
    def __add__(self,other):
        if isinstance(other,GeoUnit):
            return GeoUnit(self.value+other.value)
        elif isinstance(other,GeoArray):
            return GeoArray(self.value+other.data)
        else:
            raise TypeError("Unsupported type", other, type(other))
            
    def __sub__(self,other):
        if isinstance(other,GeoUnit):
            return GeoUnit(self.value-other.value)
        elif isinstance(other,GeoArray):
            return GeoArray(self.value-other.data)
        else:
            raise TypeError("Unsupported type", other, type(other))
    
    def __mul__(self,other):
        if isinstance(other,Number):
            return GeoUnit(self.value*other)
        elif isinstance(other,np.ndarray):
            return GeoArray(self.value*other)
        else:
            raise TypeError("Unsupported type", other, type(other))
            
    def __truediv__(self,other):
        if isinstance(other,Number):
            return GeoUnit(self.value/other)
        elif isinstance(other,GeoUnit):
            return self.value/other.value
        elif isinstance(other,np.ndarray):
            return GeoArray(self.value/other)
        else:
            raise TypeError("Unsupported type", other, type(other))

    def __eq__(self,other):
        if isinstance(other,Number):
            return self.to_degrees() == other
        else:
            return self.value == other.value

    def __gt__(self,other):
        if isinstance(other,Number):
            return self.to_degrees() > other
        else:
            return self.value > other.value
            
class GeoArray:
    def __init__(self,data):
        self.data = np.round(data)
        
    @classmethod
    def from_degrees(self,deg_arr):
        return GeoArray(deg_arr * 3600.0 * 1e6)
        
    def to_degrees(self):
        return self.data / (3600.0*1e6)
    
    def __add__(self,other):
        if isinstance(other,GeoUnit):
            return GeoArray(self.data + other.value)
        else:
            return GeoArray(self.data + other.data)

    def __sub__(self,other):
        if isinstance(other,GeoUnit):
            return GeoArray(self.data - other.value)
        else:
            return GeoArray(self.data - other.data)
    
    def __repr__(self):
        return self.to_degrees().__repr__()

    def __eq__(self,other):
        return self.data == other.data


class GeoPoint:
    def __init__(self,lat,lon):
        '''
        types : <GeoUnit,GeoUnit>
        '''
        self.lat = lat
        self.lon = lon
        
    def __add__(self,other):
        return GeoPoint(self.lat+other.lat,self.lon+other.lon)
    
    def __sub__(self,other):
        return GeoPoint(self.lat-other.lat,self.lon-other.lon)
    
    @classmethod
    def from_degrees(self,lat,lon):
        return GeoPoint(GeoUnit.from_dms(lat),GeoUnit.from_dms(lon))
    
    def to_degrees(self):
        return self.lat.to_degrees(),self.lon.to_degrees()

    def __repr__(self): 
        return("%s,%s" % (self.lat.__repr__(),self.lon.__repr__()))

    def __eq__(self,other):
        return self.lat == other.lat and self.lon == other.lon

class GeoReferenceMode(Enum):
    CENTER = 1
    CORNER = 2

class GeoReference:
    def __init__(self,origin,nlats,nlons,cell_size,lat_orient=1,lon_orient=1,mode=GeoReferenceMode.CENTER):
        '''
        types:
        origin : <GeoUnit or tuple>
        '''

        self.origin = get_geopoint(origin)
        self.nlats = nlats
        self.nlons = nlons
        self.lat_orient = lat_orient
        self.lon_orient = lon_orient
        self.cell_size = get_geounit(cell_size)
        self.mode = mode

    def to_orient(self,lat_orient):
        if lat_orient == self.lat_orient:
            return self
        else:
            ref = self.to_mode(GeoReferenceMode.CENTER)
            new_lat = ref.origin.lat + ref.cell_size * ref.lat_orient * (ref.nlats - 1)
            return GeoReference(get_geopoint((new_lat,ref.origin.lon)),ref.nlats,ref.nlons,ref.cell_size,\
                lat_orient,ref.lon_orient,ref.mode).to_mode(self.mode)

    def to_mode(self,mode):
        if mode == self.mode:
            return self
        else:
            lat_off = self.cell_size * 0.5 * self.lat_orient
            lon_off = self.cell_size * 0.5 * self.lon_orient
            offset = GeoPoint(lat_off,lon_off)
            if mode == GeoReferenceMode.CORNER:
                new_origin = self.origin - offset
            elif mode == GeoReferenceMode.CENTER:
                new_origin = self.origin + offset
            return GeoReference(new_origin,self.nlats,self.nlons,self.cell_size,self.lat_orient,self.lon_orient,mode)

    def get_offset(self,other,enforce_int=True):
        '''
        Return the number of cells by which other is offset from this GeoReference
        '''
        other = other.to_orient(self.lat_orient).to_mode(self.mode)
        diff = other.origin - self.origin
        lat_off = diff.lat / self.cell_size * self.lat_orient
        lon_off = diff.lon / self.cell_size * self.lon_orient
        if enforce_int:
            lat_off_i = int(lat_off)
            lon_off_i = int(lon_off)
            if (lat_off == lat_off_i and lon_off == lon_off_i):
                return lat_off_i, lon_off_i
            else:
                raise Exception("Non-integer offset detected")
        return lat_off, lon_off

    def geo_for_cell(self,lat_offset,lon_offset):
        return self.origin + GeoPoint(self.cell_size * self.lat_orient * lat_offset,self.cell_size * self.lon_orient * lon_offset)

    @property
    def shape(self):
        return self.nlats,self.nlons

    @property
    def latitudes(self):
        return self.origin.lat + (self.cell_size * \
                  (np.arange(self.nlats) * self.lat_orient))
    @property
    def longitudes(self):
        return self.origin.lon + (self.cell_size * \
                  (np.arange(self.nlons) * self.lon_orient))

    def bounding_rect(self,as_degrees=True):
        cref = self.to_mode(GeoReferenceMode.CORNER)
        latmin,latmax = cref.origin.lat, cref.origin.lat + (cref.cell_size * cref.nlats * cref.lat_orient)
        lonmin,lonmax = cref.origin.lon, cref.origin.lon + (cref.cell_size * cref.nlons * cref.lon_orient)
        if as_degrees:
            return latmin.to_degrees(),latmax.to_degrees(),lonmin.to_degrees(),lonmax.to_degrees()
        return latmin,latmax,lonmin,lonmax

    def __repr__(self):
        return "%s, ncells: %sx%s, cellsize: %sx%s" % \
        (self.origin, self.nlats, self.nlons, self.cell_size*self.lat_orient, self.cell_size * self.lon_orient)

    def __eq__(self,other):
        return self.__dict__ == other.__dict__

'''
Support functions for generating ExtentFactory objects from files
'''

def load_georef_flt(fn,load_mask=True):
    import gdal
    ds = gdal.Open(fn)

    if ds is None:
        raise Exception("Could not open file", fn)

    georef,mask = get_georef_gridobj(ds)
    return (georef,mask)

def get_georef_gridobj(source,load_mask=True,grid_q = None):
    gref = source.GetGeoTransform()

    origin = GeoPoint.from_degrees(gref[3],gref[0])
    cell_size = GeoUnit.from_dms(gref[1])

    if grid_q is not None:
        if not isinstance(grid_q,GeoPoint):
            grid_q = GeoUnit.from_dms(grid_q)
        grid_q = grid_q / 2.0

        origin.lat = origin.lat.quantize(grid_q)
        origin.lon = origin.lon.quantize(grid_q)

        cell_size = cell_size.quantize(grid_q)

    nlons,nlats = source.RasterXSize, source.RasterYSize

    georef = GeoReference(origin,nlats,nlons,cell_size,lat_orient=-1,mode=GeoReferenceMode.CORNER).to_mode(GeoReferenceMode.CENTER)

    if load_mask:
        rb = source.GetRasterBand(1)
        mb = rb.GetMaskBand()
        mdata = mb.ReadAsArray()
        mask = ~mdata.astype(bool)
    else:
        mask = False

    return georef,mask


def load_georef_h5mask(fn,load_mask=True):
    import h5py
    h = h5py.File(fn)
    dims = h['dimensions']

    georef = latlon_to_georef(dims['latitude'],dims['longitude'])

    if load_mask:
        mask = h['parameters']['mask'][:] <= 0
    else:
        mask = False
    h.close()
    return georef,mask

def latlon_to_georef(lats,lons):
    origin = GeoPoint.from_degrees(lats[0],lons[0])
    cell_size = GeoUnit.from_dms(lats[1] - lats[0])
    nlats,nlons = len(lats), len(lons)

    lat_orient = 1 if cell_size > GeoUnit(0) else -1
    cell_size = cell_size * lat_orient

    return GeoReference(origin,nlats,nlons,cell_size,lat_orient=lat_orient)

def load_georef_nc(fn,load_mask=True):
    from awrams.utils.io.data_mapping import managed_dataset

    ds = managed_dataset(fn)

    if load_mask:
        e = ds.get_extent()
        georef = e.parent_ref
        mask = e.mask
        if hasattr(mask,'shape'):
            if (mask == True).all():
                mask = False
    else:
        georef = latlon_to_georef(ds.variables['latitude'][...],ds.variables['longitude'][...])
        mask = False
    return georef, mask

'''
GeoUnit helper functions
'''

def get_geounit(existing):
    if isinstance(existing,GeoUnit):
        return existing
    elif isinstance(existing,Number):
        return GeoUnit.from_dms(existing)

def get_geopoint(existing):
    if isinstance(existing,GeoPoint):
        return existing
    else:
        return GeoPoint(get_geounit(existing[0]),get_geounit(existing[1]))
    #else:
    #    return GeoPoint.from_degrees(*existing)   