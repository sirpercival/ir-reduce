# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

cdef tuple grab_params(object model)
cdef int _shift_function(double* output_coordinates, double* input_coordinates,
            int output_rank, int input_rank, double* shift_data)

cdef class CompositeModel:
    cdef public object model
    cdef public object[:] components
    cdef public str peak_name, mtype

    cdef np.ndarray[double, ndim=1] evaluate(self, double* data, int size)
    cdef list evaluate_all(self, np.ndarray[double, ndim=1] data)
    cpdef list individual(self)
    cdef copy(self)

cdef class OneDFitting:
    cdef double* data
    cdef int size

    cdef void from_1d(self, np.ndarray[double, ndim=1] oned_data)
    cdef void from_2d(self, np.ndarray[double, ndim=2] twod_data, int axis=*, str combine_type=*)
    cdef list clip(self)
    cdef int find_peak(self, str pn=*)
    cdef CompositeModel fit_peaks(self, tuple p0, str model_type=*)

cdef class Extraction:
    cdef: 
        public double[:, :] dither1, dither2, flat
        public double[:, :] reg1, reg2, tell0, diff_full, diff0
        public Py_ssize_t on_axis, off_axis
        public str mtype, file1, file2, flatfile
        public list region
        public dict distorts, extracts

    cpdef rebuild(self)
    cdef CompositeModel fit_section(self, int cen, CompositeModel model, 
                                    int width, Py_ssize_t nx, Py_ssize_t ny)
    cpdef np.ndarray[double, ndim=3] fit_trace(self, list p0, 
                                               np.ndarray[double, ndim=2] lamp, 
                                               bint lamps=*, int width=*, 
                                               bint extract=*)
    cpdef fix_distortion(self, double[:, :, :] centers)
    cdef void undistort(self)
