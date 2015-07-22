# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 00:48:54 2015

@author: gray
"""

import numpy as np
cimport numpy as np
#from astropy.io import fits

cdef class Spectrum:
    cdef public double[:] _spec, _wav

cpdef np.ndarray[double, ndim=1] replace_nans(np.ndarray[double, ndim=1] data)
cpdef object fit_to_model(object model, np.ndarray[double, ndim=1] independent,
                         np.ndarray[double, ndim=1] dependent)
cpdef fitsimage(str filename, list header)
cpdef np.ndarray[double, ndim=1] twod_to_oned(np.ndarray[double, ndim=2] array,
                                             int axis=*, str method=*)
cpdef np.ndarray[double, ndim=1] interp(np.ndarray[double, ndim=1] dependent, 
                                       np.ndarray[double, ndim=1] independent,
                                       np.ndarray[double, ndim=1] target)

cdef class ScaleableStack:
    cdef bint scaled
    cdef list _stack
    cdef void _scale(self)

cdef class ImageStack(ScaleableStack):
    cdef public list stack_list, headers, dithers, ditherpairs
    cdef public str files_card, filestring, stub
    cdef public double[:, :] _combine
    cpdef list medcombine(self, str outputfile=*)
    cpdef scale(self)

cdef class SpectrumStack(ScaleableStack):
    cdef public list headers, stack_list
    cdef double[:] _combine, _x
    cpdef scale(self, Py_ssize_t index=*)
    cpdef Spectrum combine(self, bint median=*)
    cpdef SpectrumStack subset(self, list quality)

cdef class ScalableImage:
    cdef double[:, :] _data
    cdef int nx, ny, lo, hi
    cdef double factor
    cdef str mode
    cdef list _scaled
    cpdef load(self, np.ndarray[double, ndim=2] data, str scalemode=*, 
               double factor=*)
    cdef zscale(self, float contrast=*, int num_points=*, 
                int num_per_row=*)
    cpdef change_parameters(self, dict info)
    cdef void imstretch(self)

cdef class InstrumentProfile:
    cdef public str instid, tracedir, description
    cdef public list dimensions
    cdef public dict headerkeys

cdef class ObsTarget:
    cdef public str targid, instrument_id, filestring, notes
    cdef public ImageStack images
    cdef public dict extractions
    cdef public list dither, spectra, ditherpairs

cdef class ObsNight:
    cdef public str date, filestub, rawpath, outpath, calpath, flatmethod
    cdef public dict targets
    cdef public list flaton, flatoff, cals
    cpdef bint add_to(self, ObsTarget element)
    cpdef ObsTarget get_from(self, str index)

cdef class ObsRun:
    cdef public str runid
    cdef public dict nights
    cpdef bint add_to(self, ObsNight element)
    cpdef ObsNight get_from(self, str index)

cdef class ExtractedSpectrum:
    cdef double[:] _spec, _wav
    cdef public str filename
    cdef public header
    cdef public object plot
    cpdef update_fits(self)
    cpdef save(self, str outputfile=*)