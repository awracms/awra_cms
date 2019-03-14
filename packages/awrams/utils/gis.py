from awrams.utils.metatypes import ObjectDict

from awrams.utils import extents
from awrams.utils import geo
from awrams.utils import mapping_types as mt
from awrams.utils.io.data_mapping import managed_dataset, DatasetManager
import numpy as np
import h5py

def build_transform(src_str,dest_str):
    '''
    Build a CoordinateTransformation using the supplied src and
    destination proj4 strings
    '''
    import osr
    src_ref = osr.SpatialReference()
    res = src_ref.ImportFromProj4(src_str)
    if res:
        raise Exception("Error building transformation from string %s" % src_str)

    dest_ref = osr.SpatialReference()
    res = dest_ref.ImportFromProj4(dest_str)

    if res:
        raise Exception("Error building transformation from string %s" % dest_str)

    transform = osr.CoordinateTransformation(src_ref,dest_ref)

    return transform

def polygon_from_bounds(lat0,lat1,lon0,lon1):
    from osgeo import ogr
    poly = ogr.Geometry(ogr.wkbLinearRing)
    poly.AddPoint(lon0,lat0)
    poly.AddPoint(lon0,lat1)
    poly.AddPoint(lon1,lat1)
    poly.AddPoint(lon1,lat0)
    poly.AddPoint(lon0,lat0)
    c_poly = ogr.Geometry(ogr.wkbPolygon)
    c_poly.AddGeometry(poly)
    return c_poly

def extent_to_polygon(extent):
    return polygon_from_bounds(*extent.bounding_rect())

def calc_intersection(bounds,geometry,compute_areas=True):
    '''
    Given a from_boundary_offset and an item of ogr geometry, calculate the areas
    of intersecting cells
    '''

    # +++
    # Still questionable regarding where to perform intersection
    # if inputs have different coordinate systems...
    area_data = np.zeros(bounds.shape)
    areas = np.ma.MaskedArray(data=area_data,mask=bounds.mask.copy())

    subex = extents.subdivide_extent(bounds,16)

    for e in subex:
        # poly_env = e.poly_envelope()
        poly_env = extent_to_polygon(e)
        
        if geometry.Contains(poly_env):
            for cell in e.itercells(False):
                # cpoly = cell.to_polygon()
                #from_cell_offset(cell[0],cell[1],parent_ref=e.parent_ref).to_polygon()
                lcell = cell#bounds.localise_cell(cell)

                if compute_areas:
                    cpoly = extent_to_polygon(bounds.ioffset[cell[0],cell[1]])
                    cpoly.Transform(_LONGLAT_TO_AEA)
                    areas[lcell[0],lcell[1]] = cpoly.Area()
                else:
                    areas[lcell[0],lcell[1]] = 1
        else:
            l_geom = geometry.Intersection(poly_env)
            for cell in e.itercells(False):
                # cpoly = cell.to_polygon()
                cpoly = extent_to_polygon(bounds.ioffset[cell[0],cell[1]])#from_cell_offset(cell[0],cell[1],parent_ref=e.parent_ref).to_polygon()
                #lcell = bounds.localise_cell(cell)
                lcell = cell

                if cpoly.Intersects(l_geom):
                    if compute_areas:
                        if l_geom.Contains(cpoly):
                            cpoly.Transform(_LONGLAT_TO_AEA)
                            areas[lcell[0],lcell[1]] = cpoly.Area()
                        else:
                            isect = cpoly.Intersection(l_geom)
                            isect.Transform(_LONGLAT_TO_AEA)
                            areas[lcell[0],lcell[1]] = isect.Area()
                        
                    else:
                        areas[lcell[0],lcell[1]] = 1
                else:
                    areas.mask[lcell[0],lcell[1]] = True

    return areas

def extent_from_record(record,parent_extent,compute_areas=True):
    '''
    Build an Extent from the supplied ogr shapefile record, and compute the areas covered by each cell
    '''
    geometry = record.GetGeometryRef()

    sref = geometry.GetSpatialReference()
    p4_rep = sref.ExportToProj4()

    LL_TRANSFORM = build_transform(p4_rep,_LONGLAT_PROJ)

    ll_geo = geometry.Clone()
    ll_geo.Transform(LL_TRANSFORM)

    sh_bounds = ll_geo.GetEnvelope()

    meta = {}
    for k,v in record.items().items():
        meta[k] = v
        
    e = parent_extent.factory.get_by_boundary_coords(sh_bounds[3],sh_bounds[2],sh_bounds[0],sh_bounds[1])

    areas = calc_intersection(e.translate_localise_origin(),ll_geo,compute_areas)

    e.set_mask(areas.mask)
    e.set_areas(areas)

    return e

class ShapefileDB(object):
    def __init__(self,shp_file):
        '''
        Container for records within specified shapefile
        '''
        from osgeo import ogr
        ds = ogr.Open(shp_file)

        if ds is None:
            raise Exception("Can't open shapefile from %s"%shp_file)

        layer = ds.GetLayer()

        self.records = []

        for i in range(0,layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            self.records.append(feature)

    def get_records_df(self):
        '''
        Return a pandas DataFrame of record items
        '''
        import pandas as pd
        records = ([r.items() for r in self.records])
        return pd.DataFrame(records)

    def get_extent_by_field(self,field,value,parent_extent,compute_areas=True):
        '''
        Return an Extent in the coordinates of the supplied parent extent, by matching field,value 
        '''
        for r in self.records:
            if r[field] == value:
                return extent_from_record(r,parent_extent,compute_areas)

class ExtentStoreShapefile:
    def __init__(self,shapefile,key,parent_extent):
        self.sdb = ShapefileDB(shapefile)
        self.parent_extent = parent_extent
        self.available = list(self.sdb.get_records_df()[key])
        self.key = key
        
    def __getitem__(self,k):
        return self.sdb.get_extent_by_field(self.key,k,self.parent_extent)

class ExtentStoreNC:
    def __init__(self,ncfile,mode='r'):
        if mode == 'r':
            self.f = h5py.File(ncfile,'r')
            self.available = [m for m in list(self.f) if isinstance(self.f[m],h5py.Group)]
            self._fastget = True
        else:
            self.f = managed_dataset(ncfile,mode)
            self.available = sorted(self.f.groups)
            self._fastget = False
        
    def close(self):
        self.f.close()
        
    def _getitem_fast(self,k):
        g = self.f[k]
        lats = g['latitude'][...]
        lons = g['longitude'][...]
        avar = g['area']
        areas = avar[...]
        mask = areas == 0.

        cell_size = float(avar.attrs['cell_size'])
        lat_orient = int(avar.attrs['lat_orient'])

        origin = geo.GeoPoint.from_degrees(lats[0],lons[0])
        georef = geo.GeoReference(origin,len(lats),len(lons),cell_size,lat_orient)
        return extents.Extent(georef,mask=mask,areas=areas)
        
    def __getitem__(self,k):
        if self._fastget:
            return self._getitem_fast(k)
        
        g = self.f.groups[k]
        lats = g.variables['latitude'][...]
        lons = g.variables['longitude'][...]
        avar = g.variables['area']
        areas = avar[...]
        mask = areas == 0.
        
        cell_size = float(avar.getncattr('cell_size'))
        lat_orient = int(avar.getncattr('lat_orient'))
        
        origin = geo.GeoPoint.from_degrees(lats[0],lons[0])
        georef = geo.GeoReference(origin,len(lats),len(lons),cell_size,lat_orient)
        return extents.Extent(georef,mask=mask,areas=areas)
    
    def __setitem__(self,k,extent):
        group = self.f.create_group(k)
        g = DatasetManager(group)
        lats = mt.Coordinates(mt.latitude,extent.latitudes.to_degrees())
        lons = mt.Coordinates(mt.longitude,extent.longitudes.to_degrees())
        cs = mt.CoordinateSet([lats,lons])
        v = mt.Variable('area','m2',dict(long_name='Metres squared covered by cell',cell_size=extent.cell_size.to_degrees(),lat_orient=extent.parent_ref.lat_orient))
        mapped_var = mt.MappedVariable(v,cs,np.float64)
        ncv = g.create_variable(mapped_var)
        ncv[...] = extent.areas.data


#Default projection, as per Andrew's R script
_LONGLAT_TO_AEA = None
try:
    _AEA_AUS_PROJ = "+proj=aea +ellps=GRS80, +lat_1=-18.0 +lat_2=-36.0 +units=m +lon_0=134.0 +pm=greenwich"
    _LONGLAT_PROJ = '+proj=longlat +ellps=GRS80 +no_defs'
    _LONGLAT_TO_AEA = build_transform(_LONGLAT_PROJ,_AEA_AUS_PROJ)
except:
    print("WARNING: osr not available, cell area calculations will be approximate")
    pass

def degrees_to_radians(d):
    return d / 180. * np.pi

def radius(latitude):
    a = 6378137.0         ### equatorial radius GRS80
    b = 6356752.314140347 ### polar radius GRS80
    l = degrees_to_radians(latitude)
    cos = np.cos(l)
    sin = np.sin(l)
    return np.sqrt(((a**2*cos)**2 + (b**2*sin)**2) / ((a*cos)**2 + (b*sin)**2))

def calc_area(corners,ref_lat):
    """
    http://gis.stackexchange.com/questions/711/how-can-i-measure-area-from-geographic-coordinates
    http://trac.osgeo.org/openlayers/browser/trunk/openlayers/lib/OpenLayers/Geometry/LinearRing.js?rev=10116#L233
    http://trs-new.jpl.nasa.gov/dspace/bitstream/2014/40409/3/JPL%20Pub%2007-3%20%20w%20Errata.pdf

    :param corners:
    :param ref_lat:
    :return:
    """
    area = 0.0
    lr = radius(ref_lat)

    for i in range(len(corners)-1):
        p1 = corners[i]
        p2 = corners[i+1]
        area += degrees_to_radians(p2[1] - p1[1]) * (2 + np.sin(degrees_to_radians(p1[0])) + np.sin(degrees_to_radians(p2[0])))

    return area * lr**2 / 2.0

def calculate_areas(extent):
    areas = np.empty(extent.shape)

    lats = extent.latitudes.to_degrees()
    halfcell = (extent.cell_size * 0.5).to_degrees()

    if _LONGLAT_TO_AEA is not None:
        for i, lat in enumerate(lats):
            cpoly = polygon_from_bounds(lat-halfcell,lat+halfcell,134.0-halfcell,134.0+halfcell)
            cpoly.Transform(_LONGLAT_TO_AEA)
            a = cpoly.Area()
            areas[i] = a
    else:
        for i, lat in enumerate(lats):
            p = (134.0-halfcell,134.0+halfcell,lat-halfcell,lat+halfcell)
            corners = ((p[2],p[0]),(p[3],p[0]),(p[3],p[1]),(p[2],p[1]),(p[2],p[0]))
            a = calc_area(corners,lat)
            areas[i] = a

    areas[extent.mask] = 0.0

    return np.ma.MaskedArray(areas,extent.mask)