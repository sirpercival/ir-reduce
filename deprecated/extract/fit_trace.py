# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:28:21 2015

@author: gray
"""

import numpy as np
from datatypes import Robust2D
from composite_model import composite_model


def fit_trace(data, p_init, tracedir=-1, width=10, 
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
    return trace_centers[pos,:], trace_centers[~pos,:]