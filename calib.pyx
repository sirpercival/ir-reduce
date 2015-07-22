# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:36:15 2015

@author: gray
"""

import numpy as np
cimport numpy as np
cimport cython

from scipy.interpolate import interp1d
from astropy.modeling import models, fitting
from datatypes cimport replace_nans
from scipy.signal import medfilt

cdef object fitmethod = fitting.LevMarLSQFitter()

cpdef list synth_from_range(dict lines, int ncal, list sr):
    cdef double sd = sr[1] - sr[0], su = sr[1] + sr[0]
    return synthetic_spectrum(lines, ncal, 0.5*ncal*su/sd)

cpdef list centroids(np.ndarray[double, ndim=1] data_x, 
                     np.ndarray[double, ndim=1] data_y, 
                     tuple points, float wid=3.):
    cdef:
        list out = [], p0
        double p
        int n = data_x.size
        np.ndarray[double, ndim=1] window, x0 = np.arange(n, dtype=np.float64)
        np.ndarray[np.uint8_t, ndim=1, cast=True] ok
        object inter0 = interp1d(data_x, x0), inter1 = interp1d(data_x, data_y)
        object inter_x = interp1d(x0, data_x), inter_y = interp1d(x0, data_y)
        object gauss

    for i, p in enumerate(points):
        window = np.linspace(-10, 10, dtype=np.float64) + inter0(p)
        ok = np.logical_and(np.greater_equal(window, x0.min()), 
                            np.less_equal(window, x0.max()))
        window = window[ok]
        p0 = [inter1(p), window.mean(), wid]
        gauss = models.Gaussian1D(*p0) + models.Polynomial1D(2)
        gauss = fitmethod(gauss, window, inter_y(window))
        out.append(float(inter_x(gauss.mean_0.value)))
    return out

cpdef np.ndarray[double, ndim=1] normalize(np.ndarray[double, ndim=1] arr):
    cdef np.ndarray[double, ndim=1] out = replace_nans(arr)
    out -= out.min()
    out /= out.max()
    return out

cpdef np.ndarray[double, ndim=1] detrend(np.ndarray[double, ndim=1] arr, int wid=51):
    return normalize(arr - medfilt(arr, wid))

cpdef np.ndarray[double, ndim=1] c_correlate(np.ndarray[double, ndim=1] x,
                                            np.ndarray[double, ndim=1] y,
                                            np.ndarray[double, ndim=1] lag):
    '''a port of IDL's c_correlate'''
    cdef:
        np.ndarray[double, ndim=1] pxy = np.zeros_like(lag)
        double xbar = x.mean(), ybar = y.mean(), l
        double denom = np.sqrt(np.sum(np.power(x - xbar, 2)) * np.sum(np.power(y - ybar, 2)))
        Py_ssize_t i, j, n = x.size
    
    for i, l in enumerate(lag):
        if l < 0:
            j = np.trunc(np.abs(l)).astype(np.intc)
            pxy[i] = np.sum((x[j:] - xbar)*(y[:n-j] - ybar)) / denom
        else:
            j = np.trunc(l).astype(np.intc)
            pxy[i] = np.sum((x[:n-j] - xbar)*(y[j:] - ybar)) / denom
    
    return pxy

cdef list synthetic_spectrum(dict lines, int ncal, double R):
    '''Build a synthetic line spectrum. lines should be a dictionary with 2
    equal-length entries: "wavelength" (in angstroms) and "strength".'''
    cdef:
        np.ndarray[double, ndim=1] lwav, lstr, lam, spec
        double lmax, lmin, dl, l, ns
        Py_ssize_t i
    
    lwav = np.array(lines['wavelength']) * 0.0001
    lstr = np.array(lines['strength'])
    lmax = lwav.max()
    lmin = lwav.min()
    ns = np.ceil(2 * R * (lmax - lmin) / (lmax + lmin))
    #wavelength array for line list
    lam = np.arange(ns, dtype=np.float64) / (ns - 1.) * (lmax - lmin) + lmin
    dl = (lmax + lmin) / (2. * R) #line width approx. 2.3
    spec = np.zeros_like(lam)
    #set synthetic line spectrum to correct resolution
    for i,l in enumerate(lwav):
        spec += lstr[i] * np.exp(-0.5 * np.power((lam - l) / (2.3*dl), 2))
    return [lam, spec]

cpdef object calibrate_wavelength(np.ndarray[double, ndim=1] cal, 
                                  list synth, list assignment):
    cdef:
        np.ndarray[double, ndim=1] pix = np.arange(cal.size, dtype=np.float64)
        np.ndarray[double, ndim=1] adj = detrend(cal)
        np.ndarray[double, ndim=1] wav = synth[0], syn = synth[1]
        list apix = centroids(pix, adj, assignment[0])
        list awav = centroids(wav, syn, assignment[1])
        object p_init = fitmethod(models.Polynomial1D(2), apix, awav)
        np.ndarray[double, ndim=1] lam = p_init(pix)
        np.ndarray[double, ndim=1] iy = interp1d(wav, syn)(lam)
        np.ndarray[double, ndim=1] lag = np.linspace(-100, 100, num=1201, dtype=np.float64)
        np.ndarray[double, ndim=1] cc = c_correlate(adj, iy, lag)
        double ii = cc[10:-10].argmax() + 10
        object off = fitmethod(models.Polynomial1D(2), lag[ii-10:ii+10], cc[ii-10:ii+10])
        double offset = -0.5 * off.parameters[1] / off.parameters[2]
        
    apix = centroids(pix+offset, adj, tuple(apix))
    return fitmethod(p_init, apix, tuple(awav))


'''cpdef object calibrate_wavelength(np.ndarray[double, ndim=1] cal, 
                                  dict lines, list assignment, int niter=2):
    cdef: 
        np.ndarray[double, ndim=1] lam0, lwav1, ref_lambda, ref_data0
        np.ndarray[double, ndim=1] ref_data, offsets, cc, lag, lstr1
        np.ndarray[double, ndim=1] offsets_lambda, cen_pix, cen_wvl, present_data
        double offset, R
        int ncal, nx, n_cor_sec, n_pt
        object fit, p_init
        Py_ssize_t i, j, nel_per_sec, top, imax, test
        np.ndarray[Py_ssize_t, ndim=1] base_ind, ind
        list synth
    
    ncal = cal.size
    #construct an initial guess for the wavelength range from the assignment
    p_init = fitmethod(models.Polynomial1D(2), 
                       np.array([x[0] for x in assignment], dtype=np.float64),
                       np.array([x[1] for x in assignment], dtype=np.float64))
    lam0 = p_init(np.arange(ncal, dtype=np.float64))
    R = np.median(lam0) / np.fabs(np.median(np.ediff1d(lam0)))
    
    synth = synthetic_spectrum(lines, ncal, R)
    lwav1, lstr1 = synth[0], synth[1]
    
    ref_data0 = lstr1.copy()
    ref_lambda = lwav1.copy()
    present_data = cal.copy()
    
    nx = len(lam0)
    base_ind = np.arange(nx)
    n_cor_sec = 2
    nel_per_sec = np.ceil(float(nx) / float(n_cor_sec)).astype(np.intc)
    n_pt = 10 # num points used for fitting correlative max
    
    for i in xrange(niter):
        #interpolate fiducial telluric lines to wavelength array of observed telluric spectrum
        ref_data = interp1d(ref_lambda, ref_data0)(lam0)
        ref_data = np.where(np.logical_or(lam0 > np.nanmin(ref_lambda),
                                          lam0 < np.nanmax(ref_lambda)), 
                            ref_data, np.nan)
        #identify correlating positions
        #first step: match the telluric to the fiducial
        offsets = np.zeros(n_cor_sec) + np.nan
        offsets_lambda = np.zeros(n_cor_sec) + np.nan
        cen_pix = np.zeros(n_cor_sec) + np.nan
        cen_wvl = np.zeros(n_cor_sec) + np.nan
        for j in range(n_cor_sec):
            top = (j + 1) * nel_per_sec - 1
            ind = base_ind[j * nel_per_sec : np.clip(top, top, nx-1)]
            #find offset between fiducial spectrum and observed telluric
            #by finding the max of the cross-correlation
            lag = (np.arange(1201, dtype=np.float64) - 600.)/6.
            cc = c_correlate(replace_nans(present_data[ind]), 
                             replace_nans(ref_data[ind]), lag)
            imax = np.argmax(cc[n_pt:-n_pt]) + n_pt
            if imax <= n_pt or imax >= lag.size - 1 - n_pt:
                print 'ended here on iteration {}, {}'.format(i, j)
                print n_pt, lag.size, imax
                np.savetxt('/Users/gray/Desktop/cc.txt', cc)
                np.savetxt('/Users/gray/Desktop/ind.txt', ind)
                np.savetxt('/Users/gray/Desktop/ref.txt', replace_nans(ref_data))
                np.savetxt('/Users/gray/Desktop/pres.txt', replace_nans(present_data))
                np.savetxt('/Users/gray/Desktop/lag.txt', lag)
                np.savetxt('/Users/gray/Desktop/lam.txt', lam0)
                return None
            #fit the maximum of the cross-correlation function with a 2nd-degree polynomial
            p = fitmethod(models.Polynomial1D(2),
                          lag[imax - n_pt : imax + n_pt], 
                          cc[imax - n_pt : imax + n_pt])
            offset = -.5 * p.parameters[1] / p.parameters[2] #find exact center
            #store the result 
            cen_pix[j] = np.median(ind)
            cen_wvl[j] = lwav1[cen_pix[j]]
            if not ind.min() <= (cen_pix[j] + offset) <= ind.max():
                np.savetxt('/Users/gray/Desktop/cc.txt', cc)
                np.savetxt('/Users/gray/Desktop/ind.txt', ind)
                np.savetxt('/Users/gray/Desktop/ref.txt', replace_nans(ref_data))
                np.savetxt('/Users/gray/Desktop/pres.txt', replace_nans(present_data))
                np.savetxt('/Users/gray/Desktop/lag.txt', lag)
                np.savetxt('/Users/gray/Desktop/lam.txt', lam0)
                np.savetxt('/Users/gray/Desktop/wav.txt', ref_lambda)
            offsets_lambda[j] = interp1d(ind, lwav1[ind])(cen_pix[j] + offset) - \
                interp1d(ind, lwav1[ind])(cen_pix[j])
            offsets[j] = offset
        #fit the changes and apply offsets to the initial fit
        fit = fitmethod(models.Polynomial1D(1), cen_pix, offsets_lambda)
        p_init.parameters += np.array([fit.parameters[0], fit.parameters[1], 0.], dtype=np.float64)
        lam0 = p_init(np.arange(ncal, dtype=np.float64))
    
    #all done!
    print 'finished'
    return p_init #return the fit so we can save the parameters and produce a wavelength array'''
    