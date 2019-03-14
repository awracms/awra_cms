import numpy as np
import calendar
import math

q0 = 2 * math.pi * np.arange(366) / 365.
delta = 0.006918 - 0.399912 * np.cos(q0) + 0.070257 * np.sin(q0) -0.006758 * np.cos(2 * q0) + 0.000907 * np.sin(2 * q0) -0.002697 * np.cos(3 * q0) + 0.00148 * np.sin(3 * q0)
sin_delta = np.sin(delta)
cos_delta = np.cos(delta)
solar_vars = {}
rcs_distancefactor = 1+0.033*np.cos(2*math.pi*(np.arange(366)-2)/365.0)

def solar_variables(lat):
    if lat not in solar_vars:
        phi = lat * math.pi / 1.8e2
        solar_vars[lat] = (phi, np.arccos(-np.tan(phi)*np.tan(delta)))
    return solar_vars[lat]

def days_in_year(y):
    return 366 if calendar.isleap(y) else 365

def repeat_yearly(annual,start_year,end_year):
    if not end_year:
        end_year = start_year

    year_length = [days_in_year(y) for y in range(start_year,end_year+1)]
    return np.concatenate([annual[:days] for days in year_length])

def trim_series(whole_years,start_date,end_date):
    start_index = (start_date.timetuple().tm_yday-1)
    end_index = -(days_in_year(end_date.year) - end_date.timetuple().tm_yday)
    if end_index == 0:
        return whole_years[start_index:]
    else:
        return whole_years[start_index:end_index]

def rad_clear_sky(lat,start_date,end_date):
    phi,pi = solar_variables(lat)
    whole_years = repeat_yearly(94.5*rcs_distancefactor*(pi*sin_delta*np.sin(phi)+cos_delta*np.cos(phi)*np.sin(pi))/math.pi,start_date.year,end_date.year)
    return trim_series(whole_years,start_date,end_date)

def fday(lat,start_date,end_date):
    _,pi = solar_variables(lat)
    whole_years = repeat_yearly(2*pi/(2*math.pi),start_date.year,end_date.year)
    return trim_series(whole_years,start_date,end_date)
