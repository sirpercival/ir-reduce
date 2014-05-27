from astropy.modeling import functional_models as fm, fitting, \
    SummedCompositeModel, polynomial as poly
from scipy.signal import argrelextrema, medfilt
import numpy as np
from robuststats import robust_mean as robm, robust_sigma as robs, interp_nan
from copy import deepcopy
from scipy.interpolate import interp1d, griddata


posneg = {'pos':np.greater, 'neg':np.less}

def find_peaks(idata, npeak = 1, tracedir = None, pn = 'pos'):
    data = np.array(idata) #make sure we're dealing with an array
    if len(data.shape) > 1: #check for 2D array
        if traceder is None:
            tracedir = 1 #assume that the second axis is the right one...
            #if idata is 2D, compress along the trace using a robust mean
        data = robm(data, axis = tracedir)
    #since argrelextrema isn't working, just return argmax
    print pn, data.argmax(), data.argmin()
    if pn == 'pos':
        return np.nanargmax(data)
    else:
        return np.nanargmin(data)
    
    ps = zip(range(data.size), data)
    junk, data = zip(*interp_nan(ps)) #remove nans via interpolation
    data = medfilt(data, 5) #a little smoothness never hurt
    #find /all/ rel extrema:
    maxima = argrelextrema(data, posneg[pn])
    max_val = data[maxima]
    
    priority = np.argsort(-np.fabs(max_val))
    print max_val
    return maxima[priority[:npeak]]

model_types = {'Gaussian':fm.Gaussian1D, 'Lorentzian':fm.Lorentz1D}
fitmethod = fitting.NonLinearLSQFitter()

def multi_peak_model(x, amplitudes = [1.], means = [0.], sigmas = [1.], shape='Gaussian'):
    #astropy can't fit composite models, so we have to make our own
    y = np.zeros_like(x)
    mt = model_types.get(shape, model_types['Gaussian'])
    assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
    for i,a in enumerate(amplitudes):
        model = mt(a, means[i], sigmas[i])
        y += model(x)
    return y

def fit_multipeak(idata, npeak = 1, pos = None, wid = 3., ptype = 'Gaussian'):
    if pos is None:
        pos = find_peaks(idata, npeak)
    if len(pos) < npeak:
        raise ValueError('Must have a position estimate for each peak')
    else:
        npeak = [len(pos[x]) for x in pos]
    x_data = np.array(range(len(idata)))
    f = interp1d(x_data, idata, kind='cubic')
    
    amps = f(pos['pos']), f(pos['neg'])
    med = np.median(idata)
    
    #split into positive and negative so that we can differentiate between
    #the two original images
    #pmodels = []
    #for i in range(npeak[0]):
    #    pmodels.append(model_types[ptype](amps[0][i], pos['pos'][i], wid))
    #pmodel_init = SummedCompositeModel(pmodels)
    # --> New method: using Custom1D since astropy can't fit composite models
    pmodel_init = fm.custom_model_1d(multi_peak_model)
    pmodel_init.shape = ptype
    pmodel_init.amplitudes = amps[0]
    pmodel_init.means = pos['pos']
    pmodel_init.sigmas = [wid for x in xrange(len(amps[0]))]
    pdata = np.clip(idata, a_min=med, a_max=np.nanmax(idata))
    
    #nmodels = []
    #for i in range(npeak[1]):
    #    nmodels.append(model_types[ptype](amps[1][i], pos['neg'][i], wid))
    #nmodel_init = SummedCompositeModel(nmodels)
    nmodel_init = fm.custom_model_1d(multi_peak_model)
    nmodel_init.shape = ptype
    nmodel_init.amplitudes = amps[1]
    nmodel_init.means = pos['neg']
    nmodel_init.sigmas = [wid for x in xrange(len(amps[0]))]
    ndata = np.clip(idata, a_max=med, a_min=np.nanmin(idata))
    
    return x_data, fitmethod(pmodel_init, x_data, pdata), \
        fitmethod(nmodel_init, x_data, ndata)

def build_composite(custom_model):
    base_model = model_types[custom_model.shape]
    models = [base_model(custom_model.amplitudes[i], custom_model.means[i], \
        custom_model.sigmas[i]) for i in xrange(len(custom_model.means))]
    return SummedCompositeModel(models)
    
def draw_trace(idata, x_val, pfit, nfit, fixdistort = False, fitdegree = 2):
    ns = idata.shape[1]
    midpoint = ns/2
    tc1, tc2 = midpoint, midpoint + 1

    #back-convert the custom model into a composite model
    fitp = build_composite(pfit)
    fitn = build_composite(nfit)

    trace = {'pos':[np.zeros(idata.shape) for _ in fitp._transforms], \
        'neg':[np.zeros(idata.shape) for _ in fitn._transforms]}
    apertures = {'pos':[range(ns) for _ in fitp._transforms], \
        'neg':[range(ns) for _ in fitn._transforms]}

    pcur1, ncur1 = deepcopy(fitp), deepcopy(fitn)
    pcur2, ncur2 = deepcopy(fitp), deepcopy(fitn)
    down, up = True, True
    while down and up:
        #work in both directions from the middle
        if tc1 >= 0:
            lb = max(tc1 - 20, 0)
            ub = min(tc1 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            med = np.median(piece)
            pdata = np.clip(piece,a_min=med,a_max=np.nanmax(piece))
            ndata = np.clip(piece,a_min=np.nanmin(piece),a_max=med)
            if pcur1:
                pnew1 = fitmethod(pcur1, x_val, pdata)
                for i, transform in enumerate(pnew1._transforms):
                    trace['pos'][i][:,tc1] = transform(x_val)
                    apertures['pos'][i][tc1] = transform.mean
                pcur1 = pnew1
            if ncur1:
                nnew1 = fitmethod(ncur1, x_val, ndata)
                for i, transform in enumerate(nnew1._transforms):
                    trace['neg'][i][:,tc1] = transform(x_val)
                    apertures['neg'][i][tc1] = transform.mean
                ncur1 = nnew1
            tc1 -= 1
        else:
            down = False
        if tc2 < ns:
            lb = max(tc2 - 20, 0)
            ub = min(tc2 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            med = np.median(piece)
            pdata = np.clip(piece,a_min=med,a_max=np.nanmax(piece))
            ndata = np.clip(piece,a_min=np.nanmin(piece),a_max=med)
            if pcur2:
                pnew2 = fitmethod(pcur2, x_val, pdata)
                for i, transform in enumerate(pnew2._transforms):
                    trace['pos'][i][:,tc2] = transform(x_val)
                    apertures['pos'][i][tc2] = transform.mean
                pcur2 = pnew2
            if ncur2:
                nnew2 = fitmethod(ncur2, x_val, ndata)
                for i, transform in enumerate(nnew2._transforms):
                    trace['neg'][i][:,tc2] = transform(x_val)
                    apertures['neg'][i][tc2] = transform.mean
                ncur2 = nnew2
            tc2 += 1
        else:
            up = False
    
    if not fixdistort:
        return trace
    
    if pcur1:
        ap = np.array(zip(*apertures['pos']))
        #subtract off the position of each aperture
        nap, ns = ap.shape
        meds = np.median(ap, axis=1)
        meds = np.repeat(meds.reshape(nap, 1), ns, axis=1)
        ap -= meds
        #determine median offsets and fit with a polynomial
        off_x = np.median(ap, axis=0)
        pinit = poly.Polynomial1D(fitdegree)
        x_trace = np.arange(ns)
        posfit = fitmethod(pinit, x_trace, off_x)
    else: posfit = None
    
    if ncur1:
        ap = np.array(zip(*apertures['neg']))
        #subtract off the position of each aperture
        nap, ns = ap.shape
        meds = np.median(ap, axis=1)
        meds = np.repeat(meds.reshape(nap, 1), ns, axis=1)
        ap -= meds
        #determine median offsets and fit with a polynomial
        off_x = np.median(ap, axis=0)
        pinit = poly.Polynomial1D(fitdegree)
        x_trace = np.arange(ns)
        negfit = fitmethod(pinit, x_trace, off_x)
    else: negfit = None
    
    return posfit, negfit
    
    
def undistort_imagearray(imarray, fit_distortion):
    ny, nx = imarray.shape
    yp, xp = np.mgrid[0:ny, 0:nx]
    yd, xd = yp - fit_distortion(xp), xp
    return griddata((yp, xp), imarray, (yd, xd), method='cubic')
    
def extract(fmodel, imarray, telluric, pn, lamp = None):
    fac = (1, 1) if pn == 'pos' else (-1, 1)
    ny, nx = imarray.shape
    yp, xp = np.mgrid[0:ny, 0:nx]
    aps = np.zeros((nx, len(fmodel._transforms)))
    tells = np.zeros((nx, len(fmodel._transforms)))
    if lamp:
        lamps = np.zeros((nx, len(fmodel._transforms)))
    for i, transform in enumerate(fmodel._transforms):
        ytrace = transform(yp)
        toty = np.nansum(ytrace,dtype=double, axis=0)
        aps[:, i] = fac[0] * np.nansum(imarray * ytrace, axis=0, dtype=double) / toty
        tells[:, i] = fac[0] * np.nansum(telluric * ytrace, axis=0, dtype=double) / (fac[1] * toty)
        if lamp:
            lamps[:, i] = fac[0] * np.nansum(lamp * ytrace, axis=0, dtype=double) / (fac[1] * toty)
    if lamp:
        return aps, tells, lamps
    return aps, tells

    
    