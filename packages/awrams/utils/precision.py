'''
Functionality to manipulate the precision of numeric values
'''
import numpy as np

def iround(value):
    '''
    Return a rounded integer from a fp value
    '''
    return int(round(value))

def get_precision(units):
    '''
    Get the required precision (decimal points) for the given units
    '''
    return len(str(units).split('.')[-1])

def quantize(value,units,op=round,precision=None):
    '''
    Quantize to the nearest units
    '''
    if precision is None:
        precision = get_precision(units)
    return round(op(value / units) * units,precision)

def aquantize(value,units,op=np.around,precision=None):
    '''
    Quantize an array the nearest units
    '''
    if precision is None:
        precision = get_precision(units)
    return np.around(op(value / units) * units,precision)

def sanitize_cell(cell,units=0.05):
    '''
    Round a lat/lon pair to correct units
    '''
    return quantize(cell[0],units),quantize(cell[1],units)

def sanitize_geo_array(data,units=0.05):
    '''
    Round a lat/lon pair to correct units
    '''
    return aquantize(data,units)
