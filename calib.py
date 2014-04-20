import numpy as np
import json
from copy import copy
from scipy.interpolate import interp1d
from astropy.modeling import polynomial as poly, fitting

fitmethod = fitting.NonLinearLSQFitter()

def calibrate_wavelength(cal, linelist, srange, niter = 2):
    '''linelist file should be a json file with 2
    equal-length entries: "wavelength" and "strength".
    Wavelength should be in Å.'''
    
    with open(linelist) as f:
        lines = json.read(f)
    
    lwav = np.array(lines['wavelength']) * 0.0001
    lmax = np.nanmax(lwav)
    lmin = np.nanmin(lwav)
    lstr = np.array(lines['strength'])
    
    ncal = len(cal)
    
    #construct an initial guess for the wavelength range
    p_init = poly.Polynomial1D(2) #initialize to linear
    p_init.parameters = [srange[0], float(srange[1]) / float(ncal), 0]
    lam0 = p_init(np.arange(ncal))
    R = np.median(lam0) / np.fabs(np.median(lam0[0:-2] - lam0[1:-1])
    fwhm_pix = 2.3 #our line width
    #build the synthetic line spectrum
    ns = np.ceil(2 * R * (lmax - lmin) / (lmax + lmin))
    lwav1 = np.arange(ns) / (ns - 1.) * (lmax - lmin) + lmin #wavelength array for line list
    dl = (lmax + lmin) / (2. * R)
    fwhm = fwhm_pix * dl
    #set synthetic line spectrum to correct resolution
    lstr1 = [lstr[i] * np.exp(-0.5 * np.power((lwav1 - l) / fwhm, 2)) \
        for i, l in enumerate(lwav)]
    
    ref_data0 = copy(lstr1)
    ref_lambda = copy(lwav1)
    present_data = copy(cal)
    
    for i in range(niter):
        #interpolate fiducial telluric lines to wavelength array of observed telluric spectrum
        ref_data = interp1d(ref_lambda, ref_data0)(lam0)
        ref_data = np.where(lam0 > np.nanmin(ref_lambda) or \
            lam0 < np.nanmax(ref_lambda), ref_data, np.nan)
        #identify correlating positions
        nx = len(lam0)
        base_ind = np.arange(nx)
        n_cor_sec = 2
        nel_per_sec = np.ceil(float(nx) / float(n_cor_sec))
        np = 4 # num points used for fitting correlative max
        #match the telluric to the fiducial
        offsets = np.zeros(n_cor_sec) + np.nan
        offsets_lambda = np.zeros(n_cor_sec) + np.nan
        cen_pix = np.zeros(n_cor_sec) + np.nan
        cen_wvl = np.zeros(n_cor_sec) + np.nan
        for j in range(n_cor_sec):
            top = (j + 1) * nel_per_sec - 1
            ind = base_ind[j * nel_per_sec : np.clip(top, min = top, max = nx - 1)]
            #find offset between fiducial spectrum and observed telluric
            #by finding the max of the cross-correlation
            cc = np.correlate(np.nan_to_num(present_data[ind]), \
                np.nan_to_num(ref_data[ind]), mode = 'full')
            lag = np.arange(len(cc)) - len(cc)/2
            imax = np.argmax(cc)
            if imax <= np or imax >= len(ref_data) - 1 - np:
                return None
            #fit the maximum of the cross-correlation function with a 2nd-degree polynomial
            p = fitmethod(p_init, lag[imax - np : imax + np], cc[imax - np : imax + np])
            offset = -.5 * p.parameters[1] / p.parameters[2] #find exact center
            #store the result
            cen_pix[j] = np.median(ind)
            cen_wvl[j] = lwav0[cen_pix[j]]
            offsets_lambda[j] = interp1d(ind, lwav0[ind])(cen_pix[j] + offset) - \
                interp1d(ind, lwav0[ind])(cen_pix[j])
            offsets[j] = offset
        #fit the changes and apply offsets to the initial fit
        fit = fitmethod(poly.Polynomial1D(2), cen_pix, offsets_lambda)
        p_init.parameters += fit.parameters
        lam0 = p_init(np.arange(ncal))
    
    #all done!
    return p_init #return the fit so we can save the parameters and produce a wavelength array
    