# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:25:38 2015

@author: gray

"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=2] extract_aperture(np.ndarray[DTYPE_t, ndim=2] image,
                                                  np.ndarray[DTYPE_t, ndim=3] profiles,
                                                  int ax, int n_ap):
    return np.nansum(np.tile(image[...,np.newaxis],(1,1,n_ap)) * profiles,
                     axis=ax).T

@cython.boundscheck(False)
cpdef list extract(list all_profiles, np.ndarray[DTYPE_t, ndim=2] spectral_image, 
            np.ndarray[DTYPE_t, ndim=2] telluric_image,
            int tracedir=-1, bint lamps=False, object lamp=None):
    cdef int off_axis, n_ap
    cdef np.ndarray[DTYPE_t, ndim=1] stripe
    cdef np.ndarray[DTYPE_t, ndim=3] profiles
    cdef object p
    cdef list profile
    tracedir = abs(tracedir)
    off_axis = (tracedir + 1) % 2
    n_ap = len(all_profiles[0].submodel_names)
    stripe = np.arange(spectral_image.shape[off_axis])
    profiles = np.array([[p(stripe) for p in profile] 
                         for profile in all_profiles]).transpose([[0,2,1],
                                                       [2,0,1]][tracedir])
    return map(extract_aperture, [spectral_image, telluric_image] + [lamp]*lamps)

    
    