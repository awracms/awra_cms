import csv
import numpy as np
import pandas as pd

import awrams.utils.datetools as dt
from awrams.utils.helpers import sanitize_cell
import awrams.utils.extents as extents

from awrams.utils import config_manager
from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('utils')

system_profile = config_manager.get_system_profile().get_settings()

BENCHMARK_SITES = system_profile['BENCHMARKING']['BENCHMARK_SITES']
MONTHLY_REJECTION_THRESHOLD = system_profile['BENCHMARKING']['MONTHLY_REJECTION_THRESHOLD']
ANNUAL_REJECTION_THRESHOLD = system_profile['BENCHMARKING']['ANNUAL_REJECTION_THRESHOLD']

def infer_freq(df):
    if 'M' in df.index.inferred_freq:
        return 'm'
    elif 'A' in df.index.inferred_freq:
        return 'y'
    elif 'D' in df.index.inferred_freq:
        return 'd'
    else:
        return 'd'

def read_csv(csv_file):
    d = dict()
    _csv = csv.DictReader(open(csv_file,'r'))
    for row in _csv:
        id = row.pop(_csv.fieldnames[0])
        d[id] = row
    return d


def read_id_csv(id_csv):
    with open(id_csv,'r') as in_csv:
        return [line.strip() for line in in_csv.readlines()][1:]

def get_obs_sites():
    return read_csv(BENCHMARK_SITES)

def get_sites_by_ids(site_ids, site_id_type, site_ref_type, site_set_name=None, extent=None):
    """
    cross reference site ids with site meta contained is BENCHMARK_SITES csv

    :param site_ids: list of site ids
    :param site_id_type: id type for ids in site_ids (one of column names in BENCHMARK_SITES csv)
    :param site_ref_type: column name to use for id reference name
    :param site_set_name: name of set id belongs to, only required if id appears more than once in BENCHMARK_SITES csv
    :param extent: reference extent from which to return site locations; will use default if none specified
    :return: map of extents, map of site name
    """

    if extent is None:
        extent = extents.get_default_extent()

    site_idx = get_obs_sites()
    site_extents = {}
    site_name_map = {}

    for site_id in site_ids:
        ixs = [k for k in list(site_idx.keys()) if site_idx[k][site_id_type] == site_id]
        if site_set_name is not None:
            ixs = [k for k in ixs if site_idx[k]['set'] in site_set_name]
            
        if len(ixs) == 0: # site_id not found in SiteLocationsWithIndex.csv
            continue
        # use the first occurrence of pred_index
        ref = site_idx[ixs[0]]
        site_name = ref[site_ref_type]
        site_name_map[site_name] = site_id
        #site_extents[site_name] = extents.from_cell_coords(*sanitize_cell((float(ref['Y']), float(ref['X']))))
        site_extents[site_name] = extent.icoords[float(ref['Y']), float(ref['X'])]

    return site_extents, site_name_map

def get_catchments_by_ids(ids,extent=None):
    """
    get catchment extents for ids

    :param ids: list of ids
    :return: map of extents
    """
    from awrams.utils.gis import ShapefileDB
    from awrams.utils.settings import CATCHMENT_SHAPEFILE
    extent_map = {}

    if extent is None:
        extent = extents.get_default_extent()

    for idx in ids:
        try:
            extent_map[str(idx)] = ShapefileDB(CATCHMENT_SHAPEFILE).get_extent_by_field('StationID',idx,extent.geospatial_reference())
        except Exception as e:
            logger.warning(e)
            continue
    return extent_map

def extract_sites(df, sites, site_idx, period):
    data = {}

    for i,site in enumerate(sites):
        data[site] = series_for_site(df, site, site_idx[i], period)

    out_df = pd.DataFrame(data)
    return out_df

def series_for_site(df, site, site_idx, period):
    try:
        lc_columns = [x.lower() for x in df.columns]
        
        if str(site_idx).lower() in lc_columns: # referenced by Unique_ID
            col_num = lc_columns.index(str(site_idx).lower())
            return df[df.columns[col_num]][period]
        elif site in df.columns:
            return df[site][period]
        else: raise KeyError("neither %s or %s found" % (site_idx,site))
    except IndexError:
        pass

def filter_months(ts, min_month_len=None):
    from dateutil import rrule

    out_idx = pd.DatetimeIndex(data=[])
    firstday = list(rrule.rrule(rrule.MONTHLY, dtstart=ts.index[0],until=ts.index[0]))
    if len(ts.index) > 1:
        secondpart = list(rrule.rrule(rrule.MONTHLY, dtstart=ts.index[1],until=ts.index[-1]))
        alldaylist = firstday + secondpart
    else:
        alldaylist = firstday

    for month in alldaylist:
        if ts.loc[month.strftime("%m-%Y")].size >= min_month_len:
            out_idx = out_idx.union(dt.dates('%s %s' % (dt.name_of_month[month.month],month.year)))

    return out_idx

def resample_to_months_df(in_dct, how, rescale=True):
    out_dct = {}
    for col_name in list(in_dct.keys()):
        cur_data = in_dct[col_name]

        try:
            valid_idx = filter_months(valid_only(cur_data),min_month_len=MONTHLY_REJECTION_THRESHOLD)

            if how == np.sum:
                if rescale: # for included months with missing days (>28) find the mean and then sum
                    _df = cur_data[valid_idx].resample(rule='m', how=np.mean).dropna()
                    monthly_data = pd.DataFrame(_df,index=_df.index|valid_idx).fillna(method='bfill').resample(rule='m', how=np.sum)[col_name]
                else:
                    monthly_data = cur_data[valid_idx].resample(rule='m', how=how)
            else:
                monthly_data = cur_data[valid_idx].resample(rule='m', how=how)
            out_dct[col_name] = monthly_data # do it this way cause pandas was interpreting id '201001' as a time index, doh!

        except IndexError:
            # no valid data so skip this site
            continue
    if len(out_dct.keys()) == 0:
        logger.warning("No valid data for monthly freq using MONTHLY_REJECTION_THRESHOLD=%d", MONTHLY_REJECTION_THRESHOLD)
    return out_dct

def resample_to_years_df(in_dct, how, min_months=ANNUAL_REJECTION_THRESHOLD, annual_rule='A'):
    """
    expects a resampled to monthly df
    """
    out_dct = {}

    for col in in_dct.keys():
        try:
            start = in_dct[col].index[0].year
            end = in_dct[col].index[-1].year
            ts = in_dct[col].copy()

            for yr in range(start,end+1):
                if ts.loc[str(yr)][np.isnan(ts.loc[str(yr)]) == False].count() < min_months:
                    ts.loc[str(yr)] = np.nan
            out_dct[col] = ts.resample(annual_rule, how)

        except IndexError:
            out_dct[col] = None

    return out_dct

def valid_only(series,fillValue = -99.9):
    '''
    Filter out nans from a series
    '''
    return series[(np.isnan(series) | (series==fillValue)) == False]
