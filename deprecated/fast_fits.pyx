# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:18:30 2015

@author: gray
"""

import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport pow
from cpython.cobject cimport PyCObject_FromVoidPtrAndDesc

from astropy.modeling import models
from scipy.ndimage.interpolation import geometric_transform
from datatypes import RobustData, Robust2D
from image_arithmetic import im_subtract, im_minimum

@cython.boundscheck(False)
cdef object composite_model(tuple p_init, str model_type='Gaussian'):
    cdef:
        object profile, model
        tuple p
        dict profile_shape = {'gaussian': models.Gaussian1D,
                              'lorentzian': models.Lorentz1D,
                              'moffat': models.Moffat1D}
    profile = profile_shape[model_type.lower()]
    model = profile(*p_init[0])
    for p in p_init[1:]: model += profile(*p)
    return model

@cython.boundscheck(False)
#cpdef int find_peak_2d(np.ndarray[double, ndim=2] data, int tracedir=-1, 
cpdef int find_peak_2d(double[:, :] data, int tracedir=-1, 
                       str pn='pos'):
    data = Robust2D(data)
    data.replace_nans()
    tracedir = abs(tracedir) # null tracedir (-1) becomes 2nd axis
    #compress along the trace using a robust mean
    return find_peak_1d(data.combine(axis=(tracedir+1)%2), pn=pn)

@cython.boundscheck(False)
#cpdef int find_peak_1d(np.ndarray[double, ndim=1] data, str pn='pos'):
cpdef int find_peak_1d(double[:] data, str pn='pos'):
    if not isinstance(data, RobustData):
        data = RobustData(data)
    data.clipped(inplace=True)
    if pn == 'pos':
        return data.argmax()
    else:
        return data.argmin()

@cython.boundscheck(False)
#cpdef tuple fit_peaks(np.ndarray[double, ndim=1] data, 
cpdef tuple fit_peaks(double[:] data, tuple p_init, str model_type='Gaussian'):
    '''Fit a composite spectrum to the data, using the Levenberg-Marquardt
    least-squares fitter.
    
    data -> 1D cross-section
    p_init -> list of initial estimates for parameters of each peak
    '''
    cdef:
        int mean, median, stdev
        object p, pm, nm  
        list pos_models, neg_models
    
    data = RobustData(data, index=True)
    data.replace_nans()
    mean, median, stdev = data.stats()
    fit_spectrum = data.fit_to_model(composite_model(p_init, model_type))
    pos_models = [p.parameters.tolist() for p in fit_spectrum if p.amplitude > median]
    neg_models = [p.parameters.tolist() for p in fit_spectrum if p.amplitude < median]
    pm = composite_model(tuple(pos_models), model_type=model_type) if len(pos_models) > 0 else None
    nm = composite_model(tuple(neg_models), model_type=model_type) if len(neg_models) > 0 else None
    return (data.x, pm, nm)

@cython.boundscheck(False)
#cdef object fit_section(np.ndarray[double, ndim=2] data, int c, object p0, 
cdef object fit_section(double[:, :] data, int c, object p0, 
                        Py_ssize_t tracedir, int width, int nx, int ny):
    cdef:
        np.ndarray[Py_ssize_t, ndim=2] xr, yr
        Py_ssize_t[:] on_axis = np.arange(-width,width)
        object section
    
    xr = [np.tile(np.clip(on_axis+c, 0, nx-1)[..., np.newaxis], (1, ny)),
          np.tile(np.arange(nx)[..., np.newaxis], (1, on_axis.size))][tracedir]
    yr = [np.tile(np.arange(ny)[np.newaxis, ...], (on_axis.size, 1)),
          np.tile(np.clip(on_axis+c, 0, ny-1)[np.newaxis, ...], (nx, 1))][tracedir]
    section = Robust2D(data[xr, yr]).combine(axis=tracedir)
    return section.fit_to_model(p0, x=section.index)

@cython.boundscheck(False)
#cpdef tuple fit_trace(np.ndarray[double, ndim=2] data, list p_init, 
cpdef tuple fit_trace(double[:, :] data, list p_init, int tracedir=-1, 
                      int width=10, str model_type='Gaussian', bint full=False):
    '''Now we're going to fit the whole trace iteratively, working out from
    the middle to get the gaussian centroids.'''
    cdef:
        int nx, ny, nwav
        Py_ssize_t cen, c, on_axis, off_axis
        list all_models
        np.ndarray[double, ndim=2] trace_centers
        np.ndarray[np.uint8_t, ndim=1, cast=True] pos
        str peak
        object fit_profile, up, down, component
    
    nx, ny = data.shape[0], data.shape[1]
    off_axis = abs(tracedir)
    on_axis = (off_axis + 1) % 2

    n_wav = data.shape[on_axis]
    cen = n_wav/2 + (n_wav % 2)
    all_models = [None]*n_wav
    trace_centers = np.zeros((len(p_init), n_wav))
    peak = ['x_0','mean'][model_type == 'Gaussian']

    #first we grab the center section
    fit_profile = fit_section(data, cen, composite_model(tuple(p_init), 
                                                         model_type),
                                                         on_axis, width, nx, ny)       
    up, down = fit_profile.copy(), fit_profile.copy()
    all_models[cen] = fit_profile
    trace_centers[:,cen] = np.array([getattr(component, peak).value for component in fit_profile]) 
    
    for c in xrange(1, cen+1): #now work outward from the middle
        if cen + c < [nx, ny][on_axis]:
            up = fit_section(data, cen + c, up, on_axis, width, nx, ny)  
            all_models[cen+c] = up.copy()
            trace_centers[:,cen+c] = np.array([getattr(component, peak).value for component in up])
        if cen - c >= 0:
            down = fit_section(data, cen - c, down, on_axis, width, nx, ny)
            all_models[cen-c] = down.copy()
            trace_centers[:,cen-c] = np.array([getattr(component, peak).value for component in down])
    if full:
        return tuple(all_models)
    pos = np.array([x.amplitude > np.median(data) for x in all_models[cen]])
    return trace_centers[pos,:], trace_centers[~pos,:]
    

cdef int _shift_function((double* output_coordinates, double* input_coordinates,
            int output_rank, int input_rank, double* shift_data):
    cdef np.ndarray[double, ndim=1] shift = shift_data[0]
    cdef Py_ssize_t ii
    for ii in range(input_rank):
        if ii == 0: input_coordinates[ii] = output_coordinates[ii]
        else: 
            input_coordinates[ii] = output_coordinates[ii] + (shift[0] + shift[1]*output_coordinates[0] + shift[2]*pow(output_coordinates[0],2))
    return 1

cdef void _shift_destructor(void* cobject, void *shift_data):
    free(shift_data)

def undistort_function(double* shift):
    """This is the function callable from python."""
    #cdef double* shift_data = <double*>malloc(sizeof(shift))
    #shift_data[0] = shift
    return PyCObject_FromVoidPtrAndDesc(&_shift_function,
     #                                   shift_data,
                                        shift,
                                        &_shift_destructor)

@cython.boundscheck(False)
#cpdef np.ndarray[double, ndim=2] fix_distortion(np.ndarray[double, ndim=2] image,
#                                                np.ndarray[double, ndim=2] centers,
cpdef np.ndarray[double, ndim=2] fix_distortion(double[:, :] image, double[:, :] centers,
                                                int tracedir=-1):
    '''Fit the path of the trace with a polynomial, and warp the image
    back to straight.'''
    cdef object distortion
    cdef np.ndarray[double, ndim=1] centers_median, distort_params

    tracedir = abs(tracedir)
    
    centers = Robust2D(centers)
    centers_median = (centers - np.tile(centers[:,0][..., None], (1, centers.shape[1]))).combine()
    distortion = centers_median.fit_to_model(models.Polynomial1D(degree=2), 
                                             x=centers_median.index)
    distort_params = distortion.parameters
    
    if tracedir:
        image = image.T
        return geometric_transform(image, undistort_function(<double*> &distort_params.data[0])).T
    return geometric_transform(image, undistort_function(<double*> &distort_params.data[0]))

@cython.boundscheck(False)
#cpdef list extract(tuple all_profiles, np.ndarray[double, ndim=2] spectral_image, 
#            np.ndarray[double, ndim=2] telluric_image,
cpdef list extract(tuple all_profiles, double[:, :] spectral_image, 
                   double[:, :] telluric_image, Py_ssize_t tracedir=-1, 
                   bint lamps=False, object lamp=None):
    cdef:
        int n_ap
        double[:] stripe
        np.ndarray[double, ndim=3] profiles
        np.ndarray[double, ndim=2] spectra, telluric, lamp_spec
        object p, profile
        Py_ssize_t i, off_axis, on_axis
    
    off_axis = abs(tracedir)
    on_axis = (off_axis + 1) % 2
    n_ap = len(all_profiles[0].submodel_names)
    stripe = np.arange(spectral_image.shape[off_axis], dtype=np.float64)
    spectra = np.zeros((n_ap, spectral_image.shape[on_axis]), dtype=np.float64)
    telluric = np.zeros((n_ap, spectral_image.shape[on_axis]), dtype=np.float64)
    if lamps: lamp_spec=np.zeros((n_ap, spectral_image.shape[on_axis]), dtype=np.float64)
    profiles = np.array([[p(stripe) for p in profile] for profile in all_profiles])
    for i in range(n_ap):
        spectra[i,:] = np.nansum(spectral_image * profiles[:,i,:].T, axis=off_axis)
        telluric[i,:] = np.nansum(telluric_image * profiles[:,i,:].T, axis=off_axis)
        if lamps: lamp_spec[i,:] = np.nansum(lamp * profiles[:,i,:].T, axis=off_axis)

    return [spectra, telluric] + [lamp]*lamps

@cython.boundscheck(False)
#cpdef list reduce_dither_pair(np.ndarray[double, ndim=2] dither_a, 
#                              np.ndarray[double, ndim=2] dither_b, 
cpdef list reduce_dither_pair(double[:, :] dither_a, double[:, :] dither_b, 
                              list traces, int trace_direction=1, 
                              object lamp_image=None):
    '''dither_a and dither_b are two dither positions of the same source,
    already flat-fielded. traces is a list of initial guesses for trace parameters.
    trace_direction is 1 for a horizontal trace and 0 for a vertical trace.'''
    cdef bint lamps
    cdef np.ndarray[double, ndim=2] difference_image, telluric_image#, postrace, negtrace
    cdef tuple all_profiles
    lamps = lamp_image != None
    difference_image = im_subtract(dither_a, dither_b)
    #postrace, negtrace = fit_trace(difference_image, traces, 
    #                          tracedir=trace_direction)
    #dither_a = fix_distortion(dither_a, postrace, trace_direction)
    #dither_b = fix_distortion(dither_b, negtrace, trace_direction)
    #difference_image = im_subtract(dither_a, dither_b)
    all_profiles = fit_trace(difference_image, traces, 
                             tracedir=trace_direction, full=True)
    telluric_image = im_minimum(dither_a, dither_b)
    return extract(all_profiles, difference_image, telluric_image, 
                   tracedir=trace_direction, lamps=lamps, lamp=lamp_image)