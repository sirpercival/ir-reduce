# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:44:51 2015

@author: gray
"""

from astropy.modeling import models

profile_shape = {'gaussian':models.Gaussian1D,
                 'lorentzian':models.Lorentz1D,
                 'moffat':models.Moffat1D}

def composite_model(p_init, model_type='Gaussian'):
    profile = profile_shape[model_type.lower()]
    model = profile(*p_init[0])
    for p in p_init[1:]: model += profile(*p)
    return model