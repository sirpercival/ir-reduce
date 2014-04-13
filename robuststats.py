'''
    Some robust statistics routines implementing 
    sigma-clipping to remove outliers.
'''

import numpy as np
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d

def array_process(idata, sigcut, compress = False):
    '''Create the sigma-clipped data array.'''
    data = idata[:]
    scut = np.clip(sigcut, 1.0, sigcut) #if we clip too low, this will never end

    good_data = sigma_clip(data, scut, iters=None)
    if compress:
        return good_data.compressed()
    return good_data

def robust_mean(inputdata, sigcut = 3.0, axis = None):
    data = array_process(inputdata, sigcut)
    return np.mean(data, axis = axis)


def robust_sigma(inputdata, sigcut = 3.0, axis = None):
    data = array_process(inputdata, sigcut)
    return np.std(data, axis = axis)

def interp_x(points, targ_x):
    x, y = zip(*points)
    x = array(x)
    y = array(y)
    f = interp1d(x, y, kind='cubic')
    return f(targ_x)
    
def idlhash(a, b, list = False):
    c = np.outer(a,b)
    if list:
        return c.tolist()
    return c
    
    