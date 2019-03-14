'''
Provide file conversion functions for typical input data
(E.g OzNet etc)

'''

import pandas as pd
import numpy as np
from glob import glob
import os
import awrams.utils.datetools as dt
import re



sitenames = 'A1 A2 A3 A4 A5 K1 K2 K3 K4 K5 K6 K7 K8 K10 K11 K12 K13 K14 \
             M1 M2 M3 M4 M5 M6 M7 Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9 Y10 Y11 Y12 Y13'.split()

idx_30min = [4,5,6,7]
idx_20min = [2,3,4,5]

pattern = re.compile('[\s]*([^\s]+)[\s]*')

OZNET_SITE_SURFACE_LAYER_THK = dict(A1=80.,A2=80.,A3=80.,A4=80.,A5=80.,K1=80.,K2=80.,K3=80.,K4=80.,K5=80.,K6=50.,K7=50.,K8=50.,K10=50.,K11=50.,K12=50.,K13=50.,K14=50.,M1=80.,M2=80.,M3=80.,M4=80.,M5=80.,M6=80.,M7=80.,Y1=50.,Y2=50.,Y3=80.,Y4=50.,Y5=50.,Y6=50.,Y7=50.,Y8=50.,Y9=50.,Y10=50.,Y11=50.,Y12=50.,Y13=50.)

def re_consume(to_match):
    columns = []
    while 1:
        m = pattern.match(to_match)
        if m:
            columns.append(m.groups()[0])
            to_match = to_match[m.end():]
        else:
            break
    return columns

def oznet_to_df(fn,mode='30min',by_time=None,fill_values=None):
    '''
    Convert an OzNet soil moisture file to a Pandas series
    mode is '30min' or '20min' (different file layours)
    by_time: accepts either a single string of the format HH:MM (24hrs)
             for snapshotting, or a pair of such strings (the output will
             be the mean value of all times in between these (inclusive)
             e.g ('12:00','23:59') will provide the average of midday to midnight
    '''
    if fill_values is None:
        fill_values = [-99.,-99.9]

    SNAPSHOT,AVG_PERIOD = 1000,1001

    fh = open(fn)
    lines = []
    for line in fh.readlines():
        lines.append(line)

    name = lines[0].strip()
    columns = lines[1].split(',')[1:]
    n_cols = len(columns)

    time_idx = []

    sm0 = []
    sm1 = []
    sm2 = []
    sm3 = []

    prev_dt = dt.dates('1 jan 1800')[0]

    if mode == '30min':
        idx = idx_30min
    elif mode == '20min':
        idx = idx_20min
    else:
        raise Exception("Unknown mode %s", mode)

    time_mode = None

    if by_time is not None:
        if type(by_time) == str:
            time_mode = SNAPSHOT
        else:
            min_time = by_time[0]
            max_time = by_time[1]
            time_mode = AVG_PERIOD

    for line in lines[2:]:
        rec = False
        if time_mode is None:
            rec = True
        else:
            d_time = line[11:16]
            if time_mode == SNAPSHOT:
                if d_time == by_time:
                    rec = True
            elif time_mode == AVG_PERIOD:
                if d_time >= min_time and d_time <= max_time:
                    rec = True

        if rec:
            cur_dt = pd.tseries.tools.to_datetime(line[0:16],format='%d/%m/%Y %H:%M')
            if cur_dt < prev_dt:
                print("Repeated date %s, breaking" % cur_dt)
                print(line)
                break


            time_idx.append(cur_dt)

            c_vals = re_consume(line[16:])

            if len(c_vals) != n_cols:
                print("Warning: Number of columns has changed")
                print(line)
                n_cols = len(c_vals)

            try:
                sm0.append(float(c_vals[idx[0]]))
                sm1.append(float(c_vals[idx[1]]))
                sm2.append(float(c_vals[idx[2]]))
                sm3.append(float(c_vals[idx[3]]))
            except:
                print(line)
                raise

            prev_dt = cur_dt



    def nan_fill(series):
        for fill_value in fill_values:
            series[series == fill_value] = np.nan

    if time_mode == AVG_PERIOD:
        df = pd.DataFrame()
        for series, out_name in zip([sm0,sm1,sm2,sm3],['s0','ss','sd','sg']):
            ts = pd.TimeSeries(data = series,index = time_idx)
            nan_fill(ts)
            df[out_name] = ts.resample(rule='d')
    else:
        df = pd.DataFrame(index=time_idx)
        df['s0'] = sm0
        df['ss'] = sm1
        df['sd'] = sm2
        df['sg'] = sm3
        nan_fill(df)

    fh.close()

    return df

def get_by_time(df,time):
    '''
    Return the values of a Series or DataFrame from a particular time of day
    '''
    time_idx = df.index.time == time
    return df[time_idx]

def build_all_oznet(path,by_time=None,save_path=None):
    '''
    Build a complete oznet series, returning DataFrames for each of the 4 soil layers
    by_time: accepts either a single string of the format HH:MM (24hrs)
             for snapshotting, or a pair of such strings (the output will
             be the mean value of all times in between these (inclusive)
             e.g ('12:00','23:59') will provide the average of midday to midnight
    '''
    filepaths = sorted(glob(os.path.join(path,"*sm*.txt")))

    df_map = {}

    for fp in filepaths:
        fn = fp.split('/')[-1]
        sfn = fn.split('_')
        sitename = sfn[0].upper()
        mode = sfn[1]

        print(("%s" % fn))
        df_map[sitename] = oznet_to_df(fp,mode,by_time)

    s0 = pd.DataFrame()
    s1 = pd.DataFrame()
    s2 = pd.DataFrame()
    s3 = pd.DataFrame()

    sm_map = {}
    sm_map['top'] = s0
    sm_map['shallow'] = s1
    sm_map['middle'] = s2
    sm_map['deep'] = s3

    for sitename in sitenames:
        df = df_map[sitename]
        s0[sitename] = df['s0']
        s1[sitename] = df['ss']
        s2[sitename] = df['sd']
        s3[sitename] = df['sg']

    if save_path is not None:
        for k,v in list(sm_map.items()):
            out_fn = os.path.join(save_path,'sm_%s.csv' % k)
            v.to_csv(out_fn)

    return sm_map
