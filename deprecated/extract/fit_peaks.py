# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:12:16 2015

@author: gray
"""

from datatypes import RobustData
from composite_model import composite_model


def fit_peaks(data, p_init, model_type='Gaussian'):
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
            filter(lambda x: x.amplitude < median, fit_spectrum))