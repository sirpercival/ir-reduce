# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:51:03 2015

@author: gray
"""

#from ..datatypes import FitsImage
import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from find_peaks import find_peak_2d, find_peak_1d
from fit_peaks import fit_peaks
from fit_trace import fit_trace
from fix_distortion import fix_distortion
from extract import extract
from composite_model import composite_model
from image_arithmetic import im_subtract, im_minimum
import pdb

def reduce_dither_pair(dither_a, dither_b, traces, trace_direction=1, 
                       lamp_image=None):
    '''dither_a and dither_b are two dither positions of the same source,
    already flat-fielded. traces is a list of initial guesses for trace parameters.
    trace_direction is 1 for a horizontal trace and 0 for a vertical trace.'''
    #p_init = composite_model(traces, model_type='gaussian')
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
    
    
    
    
    
    
    
    
    