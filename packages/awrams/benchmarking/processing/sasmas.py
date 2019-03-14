'''
Convert SASMAS csv files to benchmarking format (a csv file for each layer containing all sites)
Input format is yearly per site csv files derived from excel spreadsheets from Newcastle Uni
SASMAS sites record data over 4 layers:
top=0-50mm; shallow=0-300mm; middle=300-600mm; deep=600-900mm
'''

import pandas as pd
import numpy as np
from glob import glob
import datetime
import os
import awrams.utils.datetools as dt
import re

PATH = '/data/cwd_awra_data/awra_test_outputs/benchmarking/sasmas/'

SITES = {"Goulburn"  :["G1","G2","G3","G4","G5","G6"],
         "Krui river":["K1","K2","K3","K4","K5","K6"],
         "Merriwa"   :["M1","M2","M3","M4","M5","M6","M7"],
         "Stanley"   :["S1","S2","S3","S4","S5","S6","S7"]}
sitenames = []
for v in list(SITES.values()):
    sitenames.extend(v)

SNAPSHOT,AVG_PERIOD = 1000,1001

def build_all_sasmas(path=PATH, by_time=None, save_path=None):
    '''
    Build a complete sasmas series, returning DataFrames for each of the 4 soil layers
    by_time:   accepts either a single string of the format HH:MM (24hrs)
               for snapshotting, or a pair of such strings (the output will
               be the mean value of all times in between these (inclusive)
               e.g ('12:00','23:59') will provide the average of midday to midnight
    save_path: output directory for layer csv files
    '''

    time_mode = None
    if by_time is not None:
        if type(by_time) == str:
            snap_time = datetime.time(**dict(list(zip(('hour','minute','second'),list(map(int,by_time.split(':')))))))
            time_mode = SNAPSHOT
        else:
            min_time = datetime.time(**dict(list(zip(('hour','minute','second'),list(map(int,by_time[0].split(':'))))))).hour
            max_time = datetime.time(**dict(list(zip(('hour','minute','second'),list(map(int,by_time[1].split(':'))))))).hour
            time_mode = AVG_PERIOD

    layers = ['top', 'shallow', 'middle', 'deep'] #, 'profile')
    columns = ('Sm 0-50','Sm-300','Sm300-600','Sm600-900') #,'0-900mm')
    df_map = {}
    for l in layers:
        df_map[l] = pd.DataFrame(dtype=np.float64)
    df_map['profile'] = pd.DataFrame(dtype=np.float64)

    for c in list(SITES.keys()): #("Krui river",): #
        for s in SITES[c]:
            df = pd.DataFrame(columns=columns, dtype=np.float64)
            files = glob(path + '/*' + s + '*.csv') #[2009|2010].csv')
            for fn in files:
                year = re.search('([0-9]{4})', fn).group(0)
                print(fn, year)
                d = pd.read_csv(fn, header=1,parse_dates=['Date/time'], index_col='Date/time') #{'Timestamp' : '<Date/time>'}, index_col = 'Timestamp')
                # some csv files contain more than 1 year so trim them
                df = df.append(d[(d.index != 'NaT')&(d.index.year == int(year))][[k for k in d.columns if 'Sm' in k]])
            df.sort_index(inplace=True)
            df.rename(columns=dict(list(zip(columns,layers))), inplace=True)

            for (l,c) in zip(layers,columns):
                # align the indexes and add site as new column
                df_map[l] = pd.DataFrame(df_map[l], index=df_map[l].index|df[l].index)
                df_ = pd.DataFrame(df[l], index=df_map[l].index|df[l].index)
                df_map[l][s] = df_
            df_map['profile'][s] = (df_map['shallow'][s] + df_map['middle'][s] + df_map['deep'][s])/3.

    layers.append('profile')
    for l in layers:
        if time_mode == AVG_PERIOD:
            index_hour = df_map[l].index.hour
            select = (min_time<=index_hour)&(index_hour<=max_time)
            df_map[l] = df_map[l][select].resample(rule='d')

        elif time_mode == SNAPSHOT:
            index_hour = df_map[l].index.hour
            index_min  = df_map[l].index.minute
            select = (snap_time.hour==index_hour)&(snap_time.minute==index_min)
            df_map[l] = df_map[l][select].resample(rule='d')

        if save_path is not None:
            out_fn = os.path.join(save_path,'sm_%s.csv' % l)
            df_map[l].to_csv(out_fn)

    return df_map



if __name__ == '__main__':
    build_all_sasmas(by_time=['21:00','23:59'], save_path='/data/cwd_awra_data/awra_test_outputs/benchmarking/sasmas/')
    #build_all_sasmas(save_path='/data/cwd_awra_data/awra_test_outputs/benchmarking/sasmas/')
    #build(by_time='22:00')
    print(sitenames)
