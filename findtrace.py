from astropy.modeling import functional_models as fm, fitting, SummedCompositeModel
from scipy.signal import argrelextrema, medfilt
import numpy as np
from robuststats import robust_mean as robm, robust_sigma as robs


posneg = {'pos':np.greater, 'neg':np.lower}

def find_peaks(idata, npeak = 1, tracedir = None, pn = 'pos'):
    if tracedir is None and len(idata.shape) > 1: #check for 2D array
        tracedir = 0 #assume that the first axis is the right one...
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

def fit_multipeak(idata, npeak = 1, pos = None, wid = 3., ptype = 'Gaussian'):
    if pos is None:
        pos = find_peaks(idata, npeak)
    if len(pos) != npeak:
        raise ValueError('Must have a position estimate for each peak')
    amps = idata[pos]
    
    models = []
    for i in range(npeak):
        models.append(model_types[ptype](amps[i], pos[i], wid))
    model_init = SummedCompositeModel(models)
    fitmethod = fitting.NonLinearLSQFitter()
    x_data = np.array(range(len(idata)))
    final_fit = fitmethod(model_init, x_data, idata)
    return final_fit