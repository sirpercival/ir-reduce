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
    return maxima[priority[:npeak]]
    
def multi_peak_model(mtype, npeak):
    #astropy can't fit composite models, so we have to make our own
    mt = {'Gaussian':fm.Gaussian1D, 'Lorentzian':fm.Lorentz1D}[mtype]
    amps, means, sigs = (np.zeros(npeak) for x in xrange(3))
    def the_model_func(x, amplitudes=amps, means=means, sigmas=sigs):
        y = np.zeros_like(x)
        assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
        for i,a in enumerate(amplitudes):
            model = mt(a, means[i], sigmas[i])
            y += model(x)
        return y
    return the_model_func

#def multi_peak_gauss(x, amplitudes = None, means = None, sigmas = None):
#    #astropy can't fit composite models, so we have to make our own
#    y = np.zeros_like(x)
#    mt = fm.Gaussian1D
#    assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
#    for i,a in enumerate(amplitudes):
#        model = mt(a, means[i], sigmas[i])
#        y += model(x)
#    return y

#def multi_peak_lorentz(x, amplitudes = None, means = None, sigmas = None):
#    #astropy can't fit composite models, so we have to make our own
#    y = np.zeros_like(x)
#    mt = fm.Lorentz1D
#    assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
#    for i,a in enumerate(amplitudes):
#        model = mt(a, means[i], sigmas[i])
#        y += model(x)
#    return y

#model_types = {'Gaussian':multi_peak_gauss, 'Lorentzian':multi_peak_lorentz}
fitmethod = fitting.NonLinearLSQFitter()

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
    #basemodel = fm.custom_model_1d(model_types[ptype])
    pbasemodel = fm.custom_model_1d(multi_peak_model(ptype, len(amps[0])))
    pmodel_init = pbasemodel(amplitudes=amps[0], means=pos['pos'], \
        sigmas = [wid for x in xrange(len(amps[0]))])
    pdata = np.clip(idata, a_min=med, a_max=np.nanmax(idata))
    print dir(pmodel_init.fixed)
    
    #nmodels = []
    #for i in range(npeak[1]):
    #    nmodels.append(model_types[ptype](amps[1][i], pos['neg'][i], wid))
    #nmodel_init = SummedCompositeModel(nmodels)
    nbasemodel = fm.custom_model_1d(multi_peak_model(ptype, len(amps[1])))
    nmodel_init = nbasemodel(amplitudes=amps[1], means = pos['neg'], \
        sigmas = [wid for x in xrange(len(amps[0]))])
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

    
    