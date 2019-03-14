'''
Provides methods for plotting data as spatial and timeseries plots
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as _np
import pandas

import awrams.utils.datetools as datetools
import awrams.utils.extents as extents
from awrams.utils.metatypes import ObjectDict as _od
from awrams.utils.metatypes import DataDict as _dd
from awrams.visualisation import layout as _layout

from .support import _sanitize_inputs,_sanitize_period,_sanitize_aggregate_method
from .support import spatial_aggregate as _spatial_aggregate

ROW_FIELD = 'source'
COLUMN_FIELD = 'variable'

def _plot_grids(grids):
    '''
    Plot the results of a query_grids() operation
    '''
    layout = _layout.DefaultSpatialGridLayout('period','variable')
    gridview = layout.generate_view(grids)
    gridview.draw()

def _plot_spatial(source,pattern=None):
    '''
    Plot spatial (gridded) data according to supplied
    pattern
    '''
    if pattern is None:
        pattern = {}
    grids = source.query_grids(pattern)
    _plot_grids(grids)

def _plot_nans(source,pattern):
    '''
    Show locations of NaN values in data
    '''
    grids = source.query_grids(pattern)
    for datum in grids.results:
        datum.data = _np.isnan(datum.data)
    layout = _layout.ShowNaNGridLayout('period','variable')
    gridview = layout.generate_view(grids)
    gridview.draw()

def _plot_filter(source,filter_fn,pattern = None):
    '''
    Plot results filtered by binary filter_fn
    e.g filter_fn = numpy.isnan
    Will highlight NaN values
    '''
    if pattern is None:
        pattern = {}
    grids = source.query_grids(pattern)
    for datum in grids.results:
        datum.data = filter_fn(datum.data)
    layout = _layout.ShowNaNGridLayout('period','variable')
    gridview = layout.generate_view(grids)
    gridview.draw()

def show_extent(extent,context):
    '''
    Show the region covered by extent, plotted against
    the supplied context (default to continental)

    :param extent: extent of interest
    :param context: extent to plot relative to, defaults to extent.default()
    :return: None
    '''

    c_ext = extent.translate(context.geospatial_reference())

    to_plot = _od()
    to_plot.extent = context
    to_plot.data = _np.ma.ones(context.shape)
    to_plot.data.mask = context.mask
    to_plot.data[c_ext.indices][c_ext.mask==False] = 2.0

    gridview = _layout.GridView(1,1)

    cmap = _layout.highlight_map((0.1,0.3,0.1),(1.0,0,0),(0.1,0.4,0.8))

    gridview[0,0] = _layout.NoBarView(to_plot,{'title': extent.__repr__(),'cmap': cmap})

    gridview.draw()

def _default_labels_single_grid(from_dict):
    labels = _od()
    labels.title = "%s, (%s)" % ( from_dict['variable'].name, from_dict['variable'].source.name )
    labels.ylabel = "%s" % (datetools.pretty_print_period(from_dict['period']))
    labels.cmap = 'RdYlBu'
    labels.units = from_dict['variable'].units
    return labels

def _trunc(x, n):
    '''Truncates/pads a float x to n decimal places without rounding'''
    slen = len('%.*f' % (n, x))
    return str(x)[:slen]

def _plot_spatial_multi(var_group,period,extent=None,**kwds):
    if not extent:
        extent = var_group.source.extent

    vg_dict = _dd()

    for v in var_group:
        aggregator = _sanitize_aggregate_method(v)

        data = aggregator(v,period,extent)

        vg_dict.add_query_item([ROW_FIELD,COLUMN_FIELD],_od(period=period,_data=data,units=v.units,extent=extent,variable=v.name,source=v.source))

    layout = _layout.DefaultSpatialGridLayout(ROW_FIELD,COLUMN_FIELD,**kwds)

    gridview = layout.generate_view(vg_dict)
    gridview.draw()
    return gridview

def get_range(inputs,extent=None,periods=None,colapse_dimension='time'):
    '''
    Return the minimum and maximum values across all periods and extents and variables
    For use when specifying a colorbar range or y-axis limits

    :param inputs: ResultVariable or VariableGroup
    :param extent: extents object
    :param periods: pandas DatetimeIndex or list of pandas DatetimeIndex's
    :param colapse_dimension: 'time' or 'spatial'
    :return: minimum,maximum
    '''

    if not type(periods) in (str,tuple):
        periods = [periods]

    abs_min = 1e10
    abs_max = -1e10
    for singleperiod in periods:

        inputs = _sanitize_inputs(inputs)

        period = _sanitize_period(singleperiod,inputs.source.period.freq)

        if not extent:
            extent = inputs.source.extent

        for v in inputs:
            if colapse_dimension == 'time':
                aggregator = _sanitize_aggregate_method(v)
            else:
                aggregator = _spatial_aggregate

            data = aggregator(v, period, extent)

            local_min = data.min()
            local_max = data.max()
            if local_min < abs_min:
                abs_min = local_min
            if local_max > abs_max:
                abs_max = local_max

    abs_min = float(_trunc(round(abs_min, 1), 1))
    abs_max = float(_trunc(abs_max + 0.1, 1))

    return abs_min, abs_max

def spatial(inputs,period,extent=None,**kwds):
    '''
    For each period calculate the minimum and max values, then
    use these extremes for consistent colorbar values,
    If the minval and maxval values are passed in, use those instead
    Use this to get plot colour consistency across data
    indicated in multiple calls
    '''

    inputs = _sanitize_inputs(inputs)

    period = _sanitize_period(period,inputs.source.period.freq)

    if not extent:
        extent = inputs.source.extent

    return _plot_spatial_multi(inputs,period,extent,**kwds)

def timeseries(inputs,period=None,extent=None,**kwds):
    '''
    Inputs is a variable or group of variables
    If extent is larger than a single cell, variables
    will be aggregated using their default method
    '''


    inputs = _sanitize_inputs(inputs)

    if period is None:
        period = inputs.source.period
    else:
        period = _sanitize_period(period, inputs.source.period.freq)

    if extent is None:
        extent = inputs.source.extent

    vg_dict = _dd()

    for v in inputs:
        data = _spatial_aggregate(v,period,extent)
        vg_dict.add_item(_od(period=period,_data=data,units=v.units,extent=extent,method='mean',variable=v.name,source=v.source))

    layout = _layout.SpatialAggregateLayout(**kwds)

    gridview = layout.generate_view(vg_dict)
    gridview.draw()
    return gridview

