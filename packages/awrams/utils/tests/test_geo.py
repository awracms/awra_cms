from awrams.utils import geo
from awrams.utils import extents
from awrams.utils import mapping_types as mt

def test_extents_geoarray():
	e = extents.get_default_extent()
	lats,lons = e.to_coords()

	lats_geo = geo.GeoArray.from_degrees(lats.index)
	lons_geo = geo.GeoArray.from_degrees(lons.index)

	print(lats.index)
	print(lats_geo.to_degrees())
	print(lats.index - lats_geo.to_degrees())

	assert((lats.index == lats_geo.to_degrees()).all())
	assert((lons.index == lons_geo.to_degrees()).all())

def test_orient():
	o1 = geo.GeoPoint.from_degrees(-12,112.0)
	o2 = geo.GeoPoint.from_degrees(-33.95,112.0)

	cell_size = 0.05

	g1 = geo.GeoReference(o1,440,10,cell_size,lat_orient=-1)
	g2 = geo.GeoReference(o2,440,10,cell_size,lat_orient=1)

	assert(g2.to_orient(-1) == g1)
	assert(g1.to_orient(1) == g2)

	assert(g1 != g2)

	assert(g1.to_orient(1).to_orient(-1) == g1)

	assert(g1 == g1.to_mode(geo.GeoReferenceMode.CORNER).to_orient(1).to_mode(geo.GeoReferenceMode.CENTER).to_orient(-1))

