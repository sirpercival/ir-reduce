from astropy.modeling import functional_models as fm, polynomial as poly, \
    SummedCompositeModel, fitting
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, medfilt
import numpy as np
from robuststats import robust_mean as robm, robust_sigma as robs, interp_nan
from copy import deepcopy
from scipy.interpolate import interp1d, griddata
from itertools import chain

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

def get_individual_params(*params):
    amplitudes = params[::3]
    means = params[1::3]
    sigmas = params[2::3]
    assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
    return amplitudes, means, sigmas

def multi_peak_model(mtype, npeak):
    mt = {'Gaussian':fm.Gaussian1D, 'Lorentzian':fm.Lorentz1D}[mtype]
    params = list(chain.from_iterable([(1., 0., 1.) for i in xrange(npeak)]))
    def the_model_func(x, *params):
        y = np.zeros_like(x)
        amplitudes, means, sigmas = get_individual_params(*params)
        for i,a in enumerate(amplitudes):
            model = mt(a, means[i], sigmas[i])
            y += model(x)
        return y
    return the_model_func

fitmethod = fitting.NonLinearLSQFitter()

def build_composite(custom_model, mtype):
    base_model = {'Gaussian':fm.Gaussian1D, 'Lorentzian':fm.Lorentz1D}[mtype]
    amp, mean, sig = get_individual_params(*custom_model)
    models = [base_model(amp[i], mean[i], sig[i]) for i in xrange(len(amp))]
    return SummedCompositeModel(models)

def deconstruct_composite(model):
    return [transform.parameters for transform in model._transforms]
    #hopefully that works; if not, we'll do this
    amp, mean, sig = [], [], []
    for transform in model._transforms:
        if isinstance(transform, fm.Gaussian1D):
            a, m, s = transform.amplitude, transform.mean, transform.stddev
        elif isinstance(transform, fm.Lorentz1D):
            a, m, s = transform.amplitude, transform.x_0, transform.fwhm
        else:
            raise Exception('Unknown model type...')
        amp.append(a)
        mean.append(m)
        sig.append(s)
    return zip(amp, mean, sig)

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
    pmodel = multi_peak_model(ptype, len(amps[0]))
    pinit = list(chain.from_iterable(zip(amps[0], pos['pos'], \
        [wid for x in xrange(len(amps[0]))])))
    pdata = np.clip(idata, a_min=med, a_max=np.nanmax(idata))
    p_fit, p_tmp = curve_fit(pmodel, x_data, pdata, pinit)
    
    nmodel = multi_peak_model(ptype, len(amps[1]))
    ninit = list(chain.from_iterable(zip(amps[1], pos['neg'], \
        [wid for x in xrange(len(amps[0]))])))
    ndata = np.clip(idata, a_max=med, a_min=np.nanmin(idata))
    n_fit, n_tmp = curve_fit(nmodel, x_data, ndata, ninit)
    
    return x_data, build_composite(p_fit, ptype), build_composite(n_fit, ptype)
    
def draw_trace(idata, x_val, pfit, nfit, fixdistort = False, fitdegree = 2, ptype = 'Gaussian'):
    ns = idata.shape[1]
    midpoint = ns/2
    tc1, tc2 = midpoint, midpoint + 1
    fitp = deconstruct_composite(pfit)
    fitn = deconstruct_composite(nfit)
    p_amp, p_mean, p_sig = get_individual_params(fitp)
    n_amp, n_mean, n_sig = get_individual_params(fitn)
    np = len(p_mean)
    nn = len(n_mean)
    
    #back-convert the custom model into a composite model
    #fitp = build_composite(pfit, ptype)
    #fitn = build_composite(nfit, ptype)

    #trace = {'pos':[np.zeros(idata.shape) for _ in fitp._transforms], \
    #    'neg':[np.zeros(idata.shape) for _ in fitn._transforms]}
    #apertures = {'pos':[range(ns) for _ in fitp._transforms], \
    #    'neg':[range(ns) for _ in fitn._transforms]}
    trace = {'pos':[np.zeros(idata.shape) for _ in p_mean], \
        'neg':[np.zeros(idata.shape) for _ in n_mean]}
    apertures = {'pos':[range(ns) for _ in p_mean], \
        'neg':[range(ns) for _ in n_mean]}

    #pcur1, ncur1 = deepcopy(fitp), deepcopy(fitn)
    #pcur2, ncur2 = deepcopy(fitp), deepcopy(fitn)
    pmodel, nmodel = multi_peak_model(ptype, np), multi_peak_model(ptype, nn)
    pcur1, ncur1 = deepcopy(pfit), deepcopy(nfit)
    pcur2, ncur2 = deepcopy(pfit), deepcopy(nfit)
    down, up = True, True
    while down or up:
        #work in both directions from the middle
        if tc1 >= 0:
            lb = max(tc1 - 20, 0)
            ub = min(tc1 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            med = np.median(piece)
            pdata = np.clip(piece,a_min=med,a_max=np.nanmax(piece))
            ndata = np.clip(piece,a_min=np.nanmin(piece),a_max=med)
            if pcur1:
                #pnew1 = fitmethod(pcur1, x_val, pdata)
                pnew1, psig1 = curve_fit(pmodel, x_val, pdata, pcur1)
                pnmodel = build_composite(pnew1, ptype)
                #for i, transform in enumerate(pnew1._transforms):
                for i, transform in enumerate(pnmodel._transforms):
                    trace['pos'][i][:,tc1] = transform(x_val)
                    apertures['pos'][i][tc1] = transform.mean
                pcur1 = pnew1
            if ncur1:
                #nnew1 = fitmethod(ncur1, x_val, ndata)
                nnew1, nsig1 = curve_fit(nmodel, x_val, ndata, ncur1)
                nnmodel = build_composite(nnew1, ptype)
                #for i, transform in enumerate(nnew1._transforms):
                for i, transform in enumerate(nnmodel._transforms):
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
                #pnew2 = fitmethod(pcur2, x_val, pdata)
                pnew2, psig2 = curve_fit(pmodel, x_val, pdata, pcur2)
                pnmodel = build_composite(pnew2, ptype)
                #for i, transform in enumerate(pnew2._transforms):
                for i, transform in enumerate(pnmodel._transforms):
                    trace['pos'][i][:,tc2] = transform(x_val)
                    apertures['pos'][i][tc2] = transform.mean
                pcur2 = pnew2
            if ncur2:
                #nnew2 = fitmethod(ncur2, x_val, ndata)
                nnew2, nsig2 = curve_fit(nmodel, x_val, ndata, ncur2)
                nnmodel = build_composite(nnew2, ptype)
                #for i, transform in enumerate(nnew2._transforms):
                for i, transform in enumerate(nnmodel._transforms):
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
        posfit
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

    
    