# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:18:45 2015

@author: gray
"""

import numpy as np
#import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from astropy.modeling import models
from scipy.ndimage.interpolation import geometric_transform

from image_arithmetic import im_subtract, im_minimum
from datatypes import RobustData, Robust2D

from fast_fits import extract

import pdb

"""profile_shape = {'gaussian':models.Gaussian1D,
                 'lorentzian':models.Lorentz1D,
                 'moffat':models.Moffat1D}

def composite_model(p_init, model_type='Gaussian'):
    profile = profile_shape[model_type.lower()]
    model = profile(*p_init[0])
    for p in p_init[1:]: model += profile(*p)
    return model"""

"""def fit_peaks(data, p_init, model_type='Gaussian'):
    '''Fit a composite spectrum to the data, using the Levenberg-Marquardt
    least-squares fitter.
    
    data -> 1D cross-section
    p_init -> list of initial estimates for parameters of each peak
    '''
    data = RobustData(data, index=True)
    data.replace_nans()
    mean, median, stdev = data.stats()
    fit_spectrum = data.fit_to_model(composite_model(p_init, model_type))
    return (data.x, filter(lambda x: x.amplitude > median, fit_spectrum),
            filter(lambda x: x.amplitude < median, fit_spectrum))"""

"""def fit_trace(data, p_init, tracedir=-1, width=10, 
              model_type='Gaussian', full=False):
    '''Now we're going to fit the whole trace iteratively, working out from
    the middle to get the gaussian centroids.'''
    nx, ny = data.shape
    tracedir = abs(tracedir)
   
    n_wav = data.shape[tracedir]
    cen = n_wav/2 + (n_wav % 2)
    all_models = [None]*n_wav
    trace_centers = np.zeros((len(p_init), n_wav))
    on_axis = np.arange(-width,width)
    peak = ['x_0','mean'][model_type == 'Gaussian']
    window = lambda c: ((np.tile(np.clip(on_axis+c, 0, nx-1)[...,np.newaxis], (1,ny)),
                         np.tile(np.arange(ny)[np.newaxis,...],(on_axis.size,1))),
                        (np.tile(np.arange(nx)[...,np.newaxis],(1,on_axis.size)), 
                         np.tile(np.clip(on_axis+c, 0, ny-1)[np.newaxis,...], (nx,1))))[tracedir]
    
    def fit_section(center, p0):
        xr, yr = window(center)
        section = Robust2D(data[xr, yr]).combine(axis=(tracedir+1)%2)
        return section.fit_to_model(p0, x=section.index)
    #first we grab the center section
    fit_profile = fit_section(cen, composite_model(p_init, model_type))       
    up, down = fit_profile.copy(), fit_profile.copy()
    all_models[cen] = fit_profile
    trace_centers[:,cen] = np.array([getattr(component, peak).value for component in fit_profile]) 
    
    for c in xrange(1, cen+1): #now work outward from the middle
        if cen + c < [nx, ny][tracedir]:
            up = fit_section(cen+c, up)
            all_models[cen+c] = up.copy()
            trace_centers[:,cen+c] = np.array([getattr(component, peak).value for component in up])
        if cen - c >= 0:
            down = fit_section(cen-c, down)
            all_models[cen-c] = down.copy()
            trace_centers[:,cen-c] = np.array([getattr(component, peak).value for component in down])
    if full:
        return all_models
    pos = np.array([x.amplitude > np.median(data) for x in all_models[cen]])
    return trace_centers[pos,:], trace_centers[~pos,:]"""

"""def fix_distortion(image, centers, tracedir=-1):
    '''Fit the path of the trace with a polynomial, and warp the image
    back to straight.'''
    tracedir = abs(tracedir)
    
    centers = Robust2D(centers)
    centers = (centers.T - centers.T[0,:]).T.combine()
    distortion = centers.fit_to_model(models.Polynomial1D(degree=2), 
                                      x=centers.index)
    def undistort(coords):
        xp, yp = coords
        if tracedir:
            xp, yp = yp, xp
        if tracedir:
            return yp - distortion(xp), xp
        return xp, yp-distortion(xp)

    return geometric_transform(image, undistort)"""

def reducedither_pair(dither_a, dither_b, traces, trace_direction=1, 
                       lamp_image=None):
    '''dither_a and dither_b are two dither positions of the same source,
    already flat-fielded. traces is a list of initial guesses for trace parameters.
    trace_direction is 1 for a horizontal trace and 0 for a vertical trace.'''
    lamps = lamp_image != None
    pdb.set_trace()
    difference_image = im_subtract(dither_a, dither_b)[1]
    postrace, negtrace = fit_trace(difference_image, traces, 
                              tracedir=trace_direction)
    dither_a = fix_distortion(dither_a, postrace, trace_direction)
    dither_b = fix_distortion(dither_b, negtrace, trace_direction)
    difference_image = im_subtract(dither_a, dither_b)[1]
    all_profiles = fit_trace(difference_image, traces, 
                             tracedir=trace_direction)
    telluric_image = im_minimum(dither_a, dither_b)[1]
    return extract(all_profiles, difference_image, telluric_image, 
                   tracedir=trace_direction, lamps=lamps, lamp=lamp_image)