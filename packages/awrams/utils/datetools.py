'''
Date Tools
==========

Provides functionality for defining and manipulating individual dates and time periods.

'''
import pandas as pd
import re
from pandas import datetime as dt
from pandas import datetools as pd_dt
from numpy import arange #pylint: disable=no-name-in-module
import numpy as np
from datetime import *

class Period:
    '''
    Thin wrapper for gapless daily periods
    '''
    def __init__(self,start,end):
        self.start = start
        self.end = end

    def __len__(self):
        return (self.end - self.start).days + 1

    def to_dti(self):
        return pd.DatetimeIndex(start=self.start,end=self.end,freq='d')

def dti_to_period(dti):
    return Period(dti[0],dti[-1])


def pretty_print_period(period):
    start = period[0]
    end = period[-1]

    same_year = False
    same_month = False
    same_day = False

    starts_month = False
    starts_year = False
    ends_month = False
    ends_year = False

    if start.year == end.year:
        same_year = True
        if start.month == end.month:
            same_month = True
            if start.day == end.day:
                same_day = True

    if start_of_month(start) == start:
        starts_month = True
    if start_of_year(start) == start:
        starts_year = True
    if end_of_month(end) == end:
        ends_month = True
    if end_of_year(end) == end:
        ends_year = True

    if same_day:
        return date_str(start)

    if starts_month and ends_month:
        if not (starts_year and ends_year):
            if same_month:
                return str_for_timestamp(start,'m')
            else:
                if same_year:
                    return "%s - %s %s" % (name_of_month[start.month], name_of_month[end.month], start.year)
                else:
                    return "%s %s - %s %s" % (name_of_month[start.month], start.year, name_of_month[end.month], end.year)

    if starts_year and ends_year:
        if same_year:
            return str(start.year)
        else:
            return "%s - %s" % (str(start.year),str(end.year))

    return "%s - %s" % (date_str(start), date_str(end))


def date_str(datelike):
    return datelike.strftime("%Y/%m/%d")

def str_for_timestamp(ts,freq):
    if freq == 'd':
        return ts.strftime("%Y/%m/%d")
    if freq == 'm':
        return "%s %s" % (name_of_month[ts.month], ts.year)
    if freq == 'a':
        return "%s" % ts.year

name_of_month = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}
pandas_months_list = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
month_map = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

tf_dict = {
    'D': "DAILY",
    'DAILY': "DAILY",
    '<DAY>': "DAILY",
    'W': "WEEKLY",
    'WEEKLY': "WEEKLY",
    'M': "MONTHLY",
    'MONTHLY': "MONTHLY",
    'A-DEC': "ANNUAL",
    'Y': "ANNUAL",
    'YEARLY': "ANNUAL",
    'A': "ANNUAL",
    'ANNUAL': "ANNUAL",
    'A-JUN': "A-JUN"
}

units_for_tf = {
    'DAILY': 'days',
    'WEEKLY': 'weeks',
    'MONTHLY': 'months',
    'ANNUAL': 'years'
}

pandas_tf_dict = {
    'DAILY': 'd',
    'WEEKLY': 'w',
    'MONTHLY': 'm',
    'ANNUAL': 'a',
    'daily': 'd',
    'weekly': 'w',
    'monthly': 'm',
    'annual': 'a',
    'A-JUN': "A-JUN",
    'days': 'd',
    'seconds': 's'
}

def timespecs_from_str(epoch_str):
    spl = epoch_str.split()
    if len(spl) == 4:
        ts_str = spl[2]+' '+spl[3]
    else:
        ts_str = spl[2]
    return pd.Timestamp(ts_str),pandas_tf_dict[spl[0]]


def filter_whole_months(index, min_month_len=None):
    '''
    return a DatetimeIndex consisting only of the whole months in the input
    '''
    start = index[0]
    cur_y = start.year
    cur_m = start.month
    cur_m_len = 1
    full_mon_len = month_len(cur_y,cur_m)
    #if min_month_len is not None:
    #    full_mon_len = min_month_len
    start_valid = (start.day == 1)
    eom = end_of_month(start)
    full_months = []
    for ts in index[1:]:
        cur_m_len += 1
        if start_valid:
            if ts == eom:
                if (cur_m_len == full_mon_len):
                    full_months.append((cur_y,cur_m))
            elif min_month_len is not None:
                if (cur_m_len >= min_month_len):
                    full_months.append((cur_y,cur_m))
        new_month = False
        if ts.year != cur_y:
            cur_y = ts.year
            new_month = True
        if ts.month != cur_m:
            cur_m = ts.month
            new_month = True
        if new_month:
            cur_m_len = 1
            start_valid = (ts.day == 1)
            eom = end_of_month(ts)
            full_mon_len = month_len(cur_y,cur_m)
            #if min_month_len is not None:
            #    full_mon_len = min_month_len

    out_idx = pd.DatetimeIndex(data=[])
    for m_pair in full_months:
        out_idx += dates('%s %s' % (name_of_month[m_pair[1]],m_pair[0]))
    return out_idx

def filter_series(series, timeframe, min_valid):
    '''
    Return a pandas series whose values are masked for any <timeframe>
    not containing at least min_valid non-nan values
    e.g 'monthly', 25 -> at least 25 valid data points per month
    '''
    tf = validate_timeframe(timeframe)
    split_p = split_period(series.index,tf)
    outseries = series.copy()
    for p in split_p:
        missing = np.isnan(series[p]).sum()
        if len(p) - missing < min_valid:
            outseries[p] = np.nan
    return outseries

def filter_series_monthly(series,thresh = 25,inverse=False):
    '''
    Special case optimized version of filter_series
    '''
    out = series.copy()
    nanidx = series.index[np.isnan(series)]
    myidx = (nanidx.year*12+(nanidx.month-1)).astype(int)
        
    nanmonths,nancounts = np.unique(myidx,return_counts=True)
    umonths = nanmonths%12
    uyears = ((nanmonths-umonths)/12).astype(int)
    umlens = [month_len(y,m) for y,m in zip(uyears,umonths+1)]

    syidx = series.index.year
    smidx = series.index.month

    out_np = np.frombuffer(out.data)
    
    if inverse:
        bad_idx = nancounts > thresh
    else:
        bad_idx = (umlens - nancounts)<thresh

    for y,m in zip(uyears[bad_idx],umonths[bad_idx]):
        yidx=np.where(syidx == y)
        midx=np.where(smidx == m+1)
        i_to_null = np.intersect1d(yidx,midx)
        out_np[i_to_null] = np.nan
        
    return out

def filter_series_annual(series,thresh = 300,inverse=False):
    '''
    Special case optimized version of filter_series
    '''
    out = series.copy()
    nanidx = series.index[np.isnan(series)]
    myidx = nanidx.year
        
    nanyears,nancounts = np.unique(myidx,return_counts=True)
    ylens = [366 if is_leap_year(y) else 365 for y in nanyears]

    syidx = series.index.year

    out_np = np.frombuffer(out.data)
    
    if inverse:
        bad_idx = nancounts > thresh
    else:
        bad_idx = (ylens - nancounts)<thresh

    for y in nanyears[bad_idx]:
        yidx=np.where(syidx == y)
        out_np[yidx] = np.nan
        
    return out
   

def month_len(year,month):
    '''
    Return the length of the given month for (year,month[1-12])
    '''
    if month in [1,3,5,7,8,10,12]:
        return 31
    elif month in [4,6,9,11]:
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    else:
        raise Exception("Unknown month %s", month)

def validate_timeframe(tf_string):
    to_tf = str(tf_string).upper()
    if to_tf in tf_dict:
        return tf_dict[to_tf]
    else:
        raise Exception("Unknown tf_string", tf_string)

class DateParseException(Exception):
    def __init__(self,date_str):
        Exception.__init__(self,"Couldn't parse date %s" % date_str)

class AmbiguousPeriodException(Exception):
    def __init__(self):
        Exception.__init__(self,"Overlapping or ambiguous period")

class EndBeforeStartException(Exception):
    def __init__(self,start_date,end_date):
        Exception.__init__(self,"End date %s before start date %s" % (end_date,start_date))

def years_for_period(period):
    '''
    Return a list of integer years for
    '''
    years = period.year
    out_years = []
    for y in years:
        if y not in out_years:
            out_years.append(y)
    return out_years

def split_discontinuous_dti(dti):
    '''
    Return [DatetimeIndex] split by any discontinuities in <dti>
    '''
    
    split_points = dti[1:] > dti[:-1] + pd.offsets.Day(1)
    split_idx = np.array(np.where(split_points == True)[0])+1
    
    out_dti = []
    
    cur_start = 0
    
    for idx in split_idx:
        out_dti.append(dti[cur_start:idx])
        cur_start = idx
    
    out_dti.append(dti[cur_start:])
    
    return out_dti

def split_period_chunks(period,chunksize):
    '''
    Split a period into chunks of at most <chunksize>
    '''
    from numpy import floor
    full_chunks = int(floor(len(period)/chunksize))
    rem = len(period)%chunksize
    slices = [slice(chunksize*k,chunksize*k+chunksize) for k in range(full_chunks)]
    if rem > 0:
        slices.append(slice(full_chunks*chunksize,len(period)))
    return [period[s] for s in slices]

def split_period(period,timeframe):
    if timeframe is None:
        return [period] #[period[0],period[-1]]

    timeframe = validate_timeframe(timeframe)
    tf_start,tf_end,tf_get = boundary_funcs(timeframe)
    if timeframe == 'MONTHLY':
        if tf_get(period[0]) == tf_get(period[-1]):
            return [period]
        else:
            cur_start = period[0]
            cur_end = tf_end(period[0])

            out_periods = []

            while cur_end < period[-1]:
                out_periods.append(dates(cur_start,cur_end))
                cur_start = cur_end + days(1)
                cur_end = tf_end(cur_start)

            cur_end = period[-1]

            out_periods.append(dates(cur_start,cur_end))

            return out_periods
    elif timeframe == 'ANNUAL':
        if period[0].year == period[-1].year:
            return [period]
        else:
            out_periods = [dates(period[0],tf_end(period[0]))]
            if period[-1].year > period[0].year + 1:
                for y in range(period[0].year + 1, period[-1].year):
                    out_periods.append(dates(y))
            out_periods.append(dates(tf_start(period[-1]),period[-1]))
            return out_periods

    elif timeframe == 'DAILY':
        if tf_get(period[0]) == tf_get(period[-1]):
            return [period]
        else:
            cur_start = period[0]
            cur_end = tf_end(period[0])

            out_periods = []

            while cur_end < period[-1]:
                out_periods.append(dates(cur_start,cur_end))
                cur_start = cur_end + days(1)
                cur_end = tf_end(cur_start)

            cur_end = period[-1]

            out_periods.append(dates(cur_start,cur_end))

            return out_periods
    else:
        raise Exception

def truncate_dti(period,to_freq):
    _ends = pd.date_range(period[0],period[-1],freq=to_freq)

    ### find to_freq period starts
    ### add S to to_freq
    tokens = to_freq.split('-')
    to_freq_start = tokens[0]+'S'
    if len(tokens) > 1: ## ie A-DEC
        to_freq_start += '-'
        to_freq_start += pandas_months_list[(pandas_months_list.index(tokens[1]) + 1)%11]

    _starts = pd.date_range(period[0],period[-1],freq=to_freq_start)
    try:
        return pd.date_range(_starts[0],_ends[-1],freq=period.freq)
    except IndexError:
        return pd.DatetimeIndex([],freq=period.freq)

def truncate_resample_dti(period,to_freq):
    if period.freq == to_freq:
        return period
    truncated = truncate_dti(period,to_freq)
    try:
        return pd.date_range(truncated[0],truncated[-1],freq=to_freq)
    except IndexError:
        return pd.DatetimeIndex([],freq=to_freq)

def truncate_partial_dti(in_period,timeframe):
    tf = validate_timeframe(timeframe)
    if tf == 'DAILY':
        return in_period
    elif tf == 'MONTHLY':
        tf_start = start_of_month
        tf_end = end_of_month
    elif tf == 'ANNUAL':
        tf_start = start_of_year
        tf_end = end_of_year
    else:
        ### already truncated
        return pd.DatetimeIndex(start=in_period[0],end=in_period[-1],freq=in_period.freq)
        # raise Exception

    in_start = in_period[0]
    in_end = in_period[-1]

    if in_start == tf_start(in_start):
        out_start = in_start
    else:
        out_start = tf_end(in_start)+days(1)
    if in_end == tf_end(in_end):
        out_end = in_end
    else:
        out_end = tf_start(in_end)-days(1)

    return pd.DatetimeIndex(start=out_start,end=out_end,freq=in_period.freq)

def resample_dti(in_period,to_timeframe,truncate=True,as_period=True):
    to_timeframe = validate_timeframe(to_timeframe)
    from_timeframe = validate_timeframe(in_period.freq)

    if to_timeframe == from_timeframe:
        return in_period

    if isinstance(in_period,pd.PeriodIndex):
        period = dates(in_period[0].start_time,in_period[-1].end_time)
    else:
        period = in_period

    if truncate:
        period = truncate_partial_dti(period,to_timeframe)
    if len(period) == 0:
        out_dti = pd.DatetimeIndex([],freq=pandas_tf_dict[to_timeframe])
    else:
        out_dti = pd.DatetimeIndex(start=period[0],end=period[-1],freq=pandas_tf_dict[to_timeframe])
    if as_period:
        out_dti = out_dti.to_period()

    return out_dti

def boundary_funcs(timeframe):
    timeframe = validate_timeframe(timeframe)
    if timeframe == 'MONTHLY':
        tf_start = start_of_month
        tf_end = end_of_month
        def tf_get(datelike):
            return datelike.year, datelike.month
        return tf_start,tf_end,tf_get
    elif timeframe == 'DAILY':
        def tf_get(datelike):
            return datelike.year, datelike.month, datelike.day
        return today,today,tf_get
    elif timeframe == 'ANNUAL':
        tf_start = start_of_year
        tf_end = end_of_year
        def tf_get(datelike):
            return datelike.year
        return tf_start,tf_end,tf_get
    elif timeframe == 'A-JUN':
        tf_start = lambda x: pd.Timestamp(datetime(x.year,7,1))
        tf_end = lambda x: pd.Timestamp(datetime(x.year+1,6,30))
        def tf_get(datelike):
            return datelike.year
        return tf_start,tf_end,tf_get

def boundaries(period,tf):
    '''
    Return start/end inclusive  boundaries for a list of days
    '''
    tf_start,tf_end,_ = boundary_funcs(tf)
    boundaries = []
    for ts in period:
        boundaries.append((tf_start(ts),tf_end(ts)))
    return boundaries


def truncate_partial_timeseries(in_series, timeframe):
    tf_start,tf_end = boundary_funcs(timeframe)

    in_start = in_series.index[0]
    in_end = in_series.index[-1]

    if in_start == tf_start(in_start):
        out_start = in_start
    else:
        out_start = tf_end(in_start)+days(1)
    if in_end== tf_end(in_end):
        out_end = in_end
    else:
        out_end = tf_start(in_end)-days(1)
    return in_series.truncate(out_start,out_end)

def days(how_many):
    '''
    Provide an offset in days
    '''
    return pd.tseries.offsets.timedelta(how_many)


def day_of_year(datelike):
    doy = datelike - start_of_year(datelike)
    return doy.days

def start_of_week(datelike):
    ts = pd.Timestamp(datelike)
    day_of_week = ts.weekday()
    return ts - pd.tseries.offsets.timedelta(day_of_week)

def today(datelike):
    return pd.Timestamp(datelike)

def start_of_month(datelike):
    ts = pd.Timestamp(datelike)
    out_ts = ts - pd.tseries.offsets.MonthBegin()
    if out_ts.month != ts.month:
        return ts
    return out_ts

def start_of_year(datelike):
    if type(datelike) == int:
        return datetime(datelike,1,1)

    ts = pd.Timestamp(datelike)
    out_ts = ts - pd.tseries.offsets.YearBegin()
    if out_ts.year != ts.year:
        return ts
    return out_ts

def end_of_week(datelike):
    ts = pd.Timestamp(datelike)
    day_of_week = ts.weekday()
    return ts + pd.tseries.offsets.timedelta(6-day_of_week)

def end_of_month(datelike):
    try:
        ts = pd.Timestamp(datelike)
    except ValueError:
        ts = pd.Timestamp(datelike.to_timestamp())

    out_ts = ts + pd.tseries.offsets.MonthEnd()
    if out_ts.month != ts.month:
        return ts
    return out_ts

def end_of_year(datelike):
    if type(datelike) == int:
        return datetime(datelike,12,31)

    ts = pd.Timestamp(datelike)
    out_ts = ts + pd.tseries.offsets.YearEnd()
    if out_ts.year != ts.year:
        return ts
    return out_ts

def is_leap_year(year):
    if year%100 == 0 and year%400 != 0:
        return False
    else:
        return year%4 == 0

def length_in_days(period):
    '''
    ++++ Hardcoded
    '''
    freq = tf_dict[period.freq]

    if freq == 'MONTHLY':
        return end_of_month(period.to_timestamp()).day
    elif freq == 'ANNUAL':
        return 366 if is_leap_year(period.year) else 365

def parse_date_str(date_str):
    '''
    Parse a string and return a period, including the resolution supplied
    '''

    def malpha_y(groups):
        month = month_map[groups[0][0:3]]
        year = int(groups[1])

        start_date = datetime(year,month,1)
        end_date = end_of_month(start_date)
        return pd.DatetimeIndex(start=start_date,end=end_date,freq='d')

    def d_malpha_y(groups):
        month = month_map[groups[1][0:3]]
        return d_m_y([groups[0],month,groups[2]])

    def malpha_d_y(groups):
        return d_malpha_y([groups[1],groups[0],groups[2]])

    def d_m_y(groups):
        return y_m_d([groups[2],groups[1],groups[0]])

    def y_m_d(groups):
        start_date = datetime(int(groups[0]),int(groups[1]),int(groups[2]))
        end_date = start_date
        return pd.DatetimeIndex(start=start_date,end=end_date,freq='d')

    def y_only(groups):
        start_date = start_of_year(int(groups[0]))
        end_date = end_of_year(int(groups[0]))
        return pd.DatetimeIndex(start=start_date,end=end_date,freq='d')

    def ranged(groups):
        start_period = parse_date_str(groups[0])
        end_period = parse_date_str(groups[1])

        return validate_ranged_period(start_period,end_period)

        return pd.DatetimeIndex(start=start_period[0],end=end_period[-1],freq='d')

    ds = date_str.lower().strip()

    patterns = dict(
        malpha_y = ["([a-z]{3,})\s*([0-9]{4})$", malpha_y],
        d_malpha_y = ["([0-9]{1,2})\s+([a-z]{3,})\s+([0-9]{4})$", d_malpha_y],
        malpha_d_y= ["([a-z]{3,})\s+([0-9]{1,2})\s+([0-9]{4})$", malpha_d_y],
        d_m_y = ["([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})$", d_m_y],
        y_m_d = ["([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$", y_m_d],
        y_only = ["([0-9]{4})$", y_only],
        ranged = ["([^-]+)-([^-]+)$", ranged]
    )

    for p_name, pattern in list(patterns.items()):
        m = re.match(pattern[0],ds)
        if m:
            return pattern[1](m.groups())

    return [pd_dt.parse_time_string(ds)[0]]

    raise DateParseException(date_str)

def validate_ranged_period(start_period,end_period):
    start_date = start_period[0]
    end_date = end_period[-1]

    if start_date > end_date:
        raise EndBeforeStartException(start_date,end_date)

    if end_period[0] <= start_period[-1]:
        if not (len(end_period) == 1 and len(start_period) == 1):
            raise AmbiguousPeriodException()

    return pd.DatetimeIndex(start=start_date,end=end_date,freq='d')

def dates(start,end=None,freq='d'):
    if type(start) == int:
        start = str(start)

    if type(start) != str:
        start_date = start
        start_period=[start_date]
    else:
        start_period = parse_date_str(start)
    
    if end is None:
        dti = pd.DatetimeIndex(start=start_period[0],end=start_period[-1],freq='d')
    else:
        if type(end) == int:
            end = str(end)

        if type(end) != str:
            end_date = end
            end_period = [end]
        else:
            end_period = parse_date_str(str(end))
            end_date = end_period[-1]

        dti = validate_ranged_period(start_period,end_period)

    if freq is not 'd':
        dti = resample_dti(dti,freq,as_period=True)
    return dti

def day(date_str):
    '''
    convenience func for single days
    '''

    p = dates(date_str)
    if len(p) > 1:
        raise Exception("Not a single date")
    else:
        return p[0]

def period_map(dti):
    '''
    Return an inverse map that returns index positions for datestamps
    '''
    return pd.Series(index=dti,data=arange(len(dti)))

def monthly_index(month_str,start_year,end_year):
    '''
    Return a PeriodIndex for the month of <month_str>, for each year in range
    '''
    return pd.PeriodIndex([pd.Period('%s %s' % (month_str, y)) for y in range(start_year,end_year+1)])

def daily_index(day_int,month_str,start_year,end_year):
    '''
    Return a PeriodIndex for the month of <month_str>, for each year in range
    '''
    return pd.PeriodIndex([pd.Period('%d %s %s' % (day_int,month_str, y)) for y in range(start_year,end_year+1)])
