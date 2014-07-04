from astropy.modeling import functional_models as fm, polynomial as poly, \
    SummedCompositeModel, fitting
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, medfilt
import numpy as np
from robuststats import robust_mean as robm, robust_sigma as robs, interp_nan
from copy import deepcopy
from scipy.interpolate import interp1d#, griddata
from scipy.ndimage.interpolation import geometric_transform
from itertools import chain
import pdb

posneg = {'pos':np.greater, 'neg':np.less}

def offset1d(reference, target):
    '''find the optimal pixel offset between reference and target
    using cross-correlation'''
    
    #The actual cross-correlation
    ycor = np.correlate(target, reference, mode='full')
    
    #Construct your pixel offset value array
    offset = np.arange(ycor.size) - (target.size - 1)
    
    #optimal offset is at the maximum of the cross-correlation
    return offset[np.nanargmax(ycor)]
    

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
    p = np.asarray(params).flatten()
    amplitudes = p[::3]
    means = p[1::3]
    sigmas = p[2::3]
    #print p
    #amplitudes = [p[0] for p in params]
    #means = [p[1] for p in params]
    #sigmas = [p[2] for p in params]
    assert len(amplitudes) == len(means) == len(sigmas), 'Parameter lists must be the same length.'
    return amplitudes, means, sigmas

def multi_peak_model(mtype, npeak):
    mt = {'Gaussian':fm.Gaussian1D, 'Lorentzian':fm.Lorentz1D}[mtype]
    params = [np.array([1., 0., 1.]) for i in xrange(npeak)]
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
    pinit = [np.array([a, pos['pos'][i], wid]) for i, a in enumerate(amps[0])]
    pdata = np.clip(idata, a_min=med, a_max=np.nanmax(idata))
    p_fit, p_tmp = curve_fit(pmodel, x_data, pdata, pinit)
    
    nmodel = multi_peak_model(ptype, len(amps[1]))
    ninit = [np.array([a, pos['neg'][i], wid]) for i, a in enumerate(amps[1])]
    ndata = np.clip(idata, a_max=med, a_min=np.nanmin(idata))
    n_fit, n_tmp = curve_fit(nmodel, x_data, ndata, ninit)
    
    return x_data, build_composite(p_fit, ptype), build_composite(n_fit, ptype)
    
def draw_trace(idata, x_val, pfit, nfit, fixdistort = False, fitdegree = 2, ptype = 'Gaussian'):
    '''move along the trace axis, fitting each position with a model of the PSF'''
    #pdb.set_trace()
    ns = idata.shape[1]
    midpoint = ns/2
    tc1, tc2 = midpoint, midpoint + 1
    fitp = deconstruct_composite(pfit)
    fitn = deconstruct_composite(nfit)
    p_amp, p_mean, p_sig = get_individual_params(*fitp)
    n_amp, n_mean, n_sig = get_individual_params(*fitn)
    n_p = len(p_mean)
    n_n = len(n_mean)
    
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

    pcur1, ncur1 = deepcopy(fitp), deepcopy(fitn)
    pcur2, ncur2 = deepcopy(fitp), deepcopy(fitn)
    pmodel, nmodel = multi_peak_model(ptype, n_p), multi_peak_model(ptype, n_n)
    #pcur1, ncur1 = deepcopy(pfit), deepcopy(nfit)
    #pcur2, ncur2 = deepcopy(pfit), deepcopy(nfit)
    down, up = True, True
    
    #set up initial data for use with cross-correlation
    piece0 = robm(idata[:, (max(tc1 - 20, 0), min(tc1 + 20, ns - 1))], axis=1)
    junk, piece0 = zip(*interp_nan(list(enumerate(piece0))))
    med = np.median(piece0)
    p0 = np.clip(piece0, a_min = med, a_max = np.nanmax(piece0))
    n0 = np.clip(piece0, a_min = np.nanmin(piece0), a_max = med)
    
    while down or up:
        #work in both directions from the middle
        if tc1 >= 0:
            lb = max(tc1 - 20, 0)
            ub = min(tc1 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            junk, piece = zip(*interp_nan(list(enumerate(piece))))
            med = np.median(piece)
            pdata = np.clip(piece,a_min=med,a_max=np.nanmax(piece))
            ndata = np.clip(piece,a_min=np.nanmin(piece),a_max=med)
            if pcur1 is not None:
                offset = offset1d(p0, pdata)
                pcur1 = [np.array([interp1d(x_val, pdata, kind='cubic', bounds_error=False)(f[1] + \
                    offset), f[1] + offset, f[2]]) for f in fitp]
                #pnew1 = fitmethod(pcur1, x_val, pdata)
                pnew1, psig1 = curve_fit(pmodel, x_val, pdata, pcur1)
                pnmodel = build_composite(pnew1, ptype)
                #for i, transform in enumerate(pnew1._transforms):
                for i, transform in enumerate(pnmodel._transforms):
                    trace['pos'][i][:,tc1] = transform(x_val)
                    apertures['pos'][i][tc1] = transform.mean
                pcur1 = pnew1
                #print tc1, pcur1
            if ncur1 is not None:
                offset = offset1d(n0, ndata)
                ncur1 = [np.array([interp1d(x_val, ndata, kind='cubic', bounds_error=False)(f[1] + \
                    offset), f[1] + offset, f[2]]) for f in fitn]
                #nnew1 = fitmethod(ncur1, x_val, ndata)
                nnew1, nsig1 = curve_fit(nmodel, x_val, ndata, ncur1)
                nnmodel = build_composite(nnew1, ptype)
                #for i, transform in enumerate(nnew1._transforms):
                for i, transform in enumerate(nnmodel._transforms):
                    trace['neg'][i][:,tc1] = transform(x_val)
                    apertures['neg'][i][tc1] = transform.mean
                ncur1 = nnew1
                #print tc1, ncur1
            tc1 -= 1
        else:
            down = False
        if tc2 < ns:
            lb = max(tc2 - 20, 0)
            ub = min(tc2 + 20, ns-1)
            piece = robm(idata[:,(lb,ub)], axis=1)
            junk, piece = zip(*interp_nan(list(enumerate(piece))))
            med = np.median(piece)
            pdata = np.clip(piece,a_min=med,a_max=np.nanmax(piece))
            ndata = np.clip(piece,a_min=np.nanmin(piece),a_max=med)
            if pcur2 is not None:
                offset = offset1d(p0, pdata)
                pcur2 = [np.array([interp1d(x_val, pdata, kind='cubic', bounds_error=False)(f[1] + \
                    offset), f[1] + offset, f[2]]) for f in fitp]
                #pnew2 = fitmethod(pcur2, x_val, pdata)
                pnew2, psig2 = curve_fit(pmodel, x_val, pdata, pcur2)
                pnmodel = build_composite(pnew2, ptype)
                #for i, transform in enumerate(pnew2._transforms):
                for i, transform in enumerate(pnmodel._transforms):
                    trace['pos'][i][:,tc2] = transform(x_val)
                    apertures['pos'][i][tc2] = transform.mean
                pcur2 = pnew2
                #print tc2, pcur2
            if ncur2 is not None:
                offset = offset1d(n0, ndata)
                ncur2 = [np.array([interp1d(x_val, ndata, kind='cubic', bounds_error=False)(f[1] + \
                    offset), f[1] + offset, f[2]]) for f in fitn]
                #nnew2 = fitmethod(ncur2, x_val, ndata)
                nnew2, nsig2 = curve_fit(nmodel, x_val, ndata, ncur2)
                nnmodel = build_composite(nnew2, ptype)
                #for i, transform in enumerate(nnew2._transforms):
                for i, transform in enumerate(nnmodel._transforms):
                    trace['neg'][i][:,tc2] = transform(x_val)
                    apertures['neg'][i][tc2] = transform.mean
                ncur2 = nnew2
                #print tc2, ncur2
            tc2 += 1
        else:
            up = False
    
    #import shelve
    #f = shelve.open('/Users/gray/Desktop/trace-shelve')
    #f['pos'] = trace['pos']
    #f['neg'] = trace['neg']
    
    if not fixdistort:
        return trace
        
    #pdb.set_trace()
    
    if pcur1 is not None:
        #identify the various aperture traces, subtract off the 
        #median x-position of each one, determine the median offset
        #across apertures, and fit with a polynomial
        if len(apertures['pos']) > 1:
            ap = np.array(zip(*apertures['pos'])).squeeze()
            ns, nap = ap.shape
            meds = np.median(ap, axis=1)
            meds = np.repeat(meds.reshape(ns, 1), nap, axis=1)
        else:
            ap = np.array(apertures['pos'][0]).squeeze()
            ns, nap = ap.size, 1
            meds = np.median(ap)
        ap -= meds
        off_x = np.median(ap, axis=0) if nap > 1 else ap
        pinit = poly.Polynomial1D(fitdegree)
        x_trace = np.arange(ns)
        posfit = fitmethod(pinit, x_trace, off_x)
        posfit
    else: posfit = None
    
    if ncur1 is not None:
        if len(apertures['neg']) > 1:
            ap = np.array(zip(*apertures['neg'])).squeeze()
            ns, nap = ap.shape
            meds = np.median(ap, axis=1)
            meds = np.repeat(meds.reshape(ns, 1), nap, axis=1)
        else:
            ap = np.array(apertures['neg'][0]).squeeze()
            ns, nap = ap.size, 1
            meds = np.median(ap)
        ap -= meds
        off_x = np.median(ap, axis=0) if nap > 1 else ap
        pinit = poly.Polynomial1D(fitdegree)
        x_trace = np.arange(ns)
        negfit = fitmethod(pinit, x_trace, off_x)
    else: negfit = None
    
    return posfit, negfit
    
    
def undistort_imagearray(imarray, fit_distortion):
    pdb.set_trace()
    
    def undistort(coords):
        yp, xp = coords
        yd, xd = yp - fit_distortion(xp), xp
        return (yd, xd)
    
    return geometric_transform(imarray, undistort)
    #ny, nx = imarray.shape
    #yp, xp = np.mgrid[0:ny, 0:nx]
    #yd, xd = yp - fit_distortion(xp), xp
    #print yp.shape, xp.shape, yd.shape, xd.shape, imarray.shape
    #pdb.set_trace()
    #return griddata((yp, xp), imarray, (yd, xd), method='cubic')
    
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

    
    