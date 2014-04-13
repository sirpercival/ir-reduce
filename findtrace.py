from astropy.modeling import functional_models as fm, fitting, \
    SummedCompositeModel, polynomial as poly
from scipy.signal import argrelextrema, medfilt
import numpy as np
from robuststats import robust_mean as robm, robust_sigma as robs
from copy import deepcopy
from scipy.interpolate import interp1d, griddata


posneg = {'pos':np.greater, 'neg':np.lower}

def find_peaks(idata, npeak = 1, tracedir = None, pn = 'pos'):
    if tracedir is None and len(idata.shape) > 1: #check for 2D array
        tracedir = 1 #assume that the second axis is the right one...
    if tracedir is not None:
        #if idata is 2D, compress along the trace using a robust mean
        data = robm(idata, axis = tracedir)
    else:
        data = idata
        
    data = medfilt(data, 5) #a little smoothness never hurt
    
    #find /all/ rel maxima:
    maxima = argrelextrema(data, posneg[pn])
    max_val = data[maxima]
    
    priority = np.argsort(max_val)
    return maxima[priority[:npeak]]

model_types = {'gauss':fm.Gaussian1D, 'lorentz':fm.Lorentz1D}
fitmethod = fitting.NonLinearLSQFitter()

def fit_multipeak(idata, npeak = 1, pos = None, wid = 3., ptype = 'Gaussian'):
    if pos is None:
        pos = find_peaks(idata, npeak)
    if len(pos) != npeak:
        raise ValueError('Must have a position estimate for each peak')
    x_data = np.array(range(len(idata)))
    f = interp1d(x_data, idata, kind='cubic')
    amps = f(pos)
    
    models = []
    for i in range(npeak):
        models.append(model_types[ptype](amps[i], pos[i], wid))
    model_init = SummedCompositeModel(models)
    return x_data, fitmethod(model_init, x_data, idata)
    
def draw_trace(idata, x_val, fitm, fixdistort = False, fitdegree = 2):
    ns = idata.shape[1]
    midpoint = ns/2
    tc1, tc2 = midpoint, midpoint + 1
    trace = np.zeros(idata.shape)
    apertures = range(ns)
    current1 = deepcopy(fitm)
    current2 = deepcopy(fitm)
    down, up = True, True
    while down and up:
        #work in both directions from the middle
        if tc1 >= 0:
            lb = max(tc1 - 20, 0)
            ub = min(tc1 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            new1 = fitmethod(current1, x_val, piece)
            trace[:,tc1] = new1(x_val)
            apertures[tc1] = [x.mean for x in new1._transforms]
            current1 = new1
            tc1 -= 1
        else:
            down = False
        if tc2 < ns:
            lb = max(tc2 - 20, 0)
            ub = min(tc2 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            new2 = fitmethod(current2, x_val, piece)
            trace[:,tc2] = new2(x_val)
            apertures[tc2] = [x.mean for x in new2._transforms]
            current2 = new2
            tc2 += 1
        else:
            up = False
    
    if not fixdistort:
        return trace
        
    apertures = np.array(zip(*apertures))
    #subtract off the position of each aperture
    nap, ns = apertures.shape
    meds = np.median(apertures, axis=1)
    meds = np.repeat(meds.reshape(nap, 1), ns, axis=1)
    apertures -= meds
    #determine median offsets and fit with a polynomial
    off_x = np.median(apertures, axis=0)
    pinit = poly.Polynomial1D(fitdegree)
    x_trace = np.arange(ns)
    pfit = fitmethod(pinit, x_trace, off_x)
    xp, yp = np.mgrid[0:nap, 0:ns]
    xd, yd = xp, yp - pfit(xp)
    fdata = griddata((xp, yp), idata, (xd, yd), method='cubic')
    return fdata
    
    
    