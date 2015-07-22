# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:42 2015

@author: gray
"""

import numpy as np
cimport numpy as np
import numpy.ma as npm
cimport cython
from cpython.cobject cimport PyCObject_FromVoidPtrAndDesc

from datatypes cimport replace_nans, fit_to_model, twod_to_oned, fitsimage
from astropy.modeling import models
from scipy.ndimage import geometric_transform
from astropy.io import fits
from astropy.stats import sigma_clip

cdef tuple grab_params(object model):
    cdef object m
    return tuple(tuple(m.parameters) for m in model)

cdef int _shift_function(double* output_coordinates, double* input_coordinates,
            int output_rank, int input_rank, double* shift_data):
    cdef np.ndarray[double, ndim=1] shift = np.asarray(<double[:3]> shift_data)
    cdef Py_ssize_t ii
    for ii in range(input_rank):
        if ii == 0: input_coordinates[ii] = output_coordinates[ii]
        else: 
            input_coordinates[ii] = output_coordinates[ii] + (shift[0] + shift[1]*output_coordinates[0] + shift[2]*pow(output_coordinates[0],2))
    return 1


def undistort_function(np.ndarray[double, ndim=1] shift):
    """This is the function callable from python."""
    cdef double* shift_data = <double*> shift.data
    return PyCObject_FromVoidPtrAndDesc(&_shift_function,
                                        shift_data,
                                        NULL)

cdef class CompositeModel:
    
    @cython.boundscheck(False)
    def __cinit__(self, tuple p_init, str model_type='Gaussian'):
        cdef:
            tuple p
            Py_ssize_t i
            object profile = {'gaussian': models.Gaussian1D,
                              'lorentzian': models.Lorentz1D,
                              'moffat': models.Moffat1D}[model_type.lower()]
            object prof
        
        self.components = np.empty(len(p_init), dtype=object)
        self.peak_name = ['x_0','mean'][model_type == 'Gaussian']
        self.mtype = model_type
        prof = profile(*p_init[0])
        self.model = prof
        self.components[0] = prof
        if len(p_init) == 1:
            return
        for i, p in enumerate(map(tuple,p_init[1:])):
            prof = profile(*p)
            self.components[i+1] = prof
            self.model += prof
    
    @cython.boundscheck(False)
    def __call__(self, np.ndarray[double, ndim=1] data):
        cdef double* data_pointer = <double*> data.data
        return self.evaluate(data_pointer, data.size)
    
    @cython.boundscheck(False)
    cdef np.ndarray[double, ndim=1] evaluate(self, double* data, int size):
        return self.model(np.asarray(<double[:size]> data))
    
    @cython.boundscheck(False)
    cdef list evaluate_all(self, np.ndarray[double, ndim=1] data):
        cdef object model
        return [model(data) for model in self.components]
    
    @cython.boundscheck(False)
    cpdef list individual(self):
        return np.asarray(self.components).tolist()
    
    property params:
        def __get__(self):
            return grab_params(self.components)
    
    @cython.boundscheck(False)
    cdef copy(self):
        return CompositeModel(grab_params(self.model), self.mtype)
    

cdef class OneDFitting:

    @cython.boundscheck(False)
    cdef void from_1d(self, np.ndarray[double, ndim=1] oned_data):
        self.data = <double*> oned_data.data
        self.size = oned_data.size
    
    @cython.boundscheck(False)
    cdef void from_2d(self, np.ndarray[double, ndim=2] twod_data, int axis=0, str combine_type='median'):
        cdef np.ndarray[double, ndim=1] temp = twod_to_oned(twod_data, axis=axis, method=combine_type)
        self.data = <double*> temp.data
        self.size = temp.size

    @cython.boundscheck(False)
    cdef list clip(self):
        cdef:
            int i=-1
            np.ndarray[np.uint8_t, ndim=1, cast=True] mask = np.ones(self.size, dtype=np.uint8)
            long lastrej = mask.sum() + 1
            np.ndarray[double, ndim=1] adj, data = np.asarray(<double[:self.size]> self.data)
        while mask.sum() != lastrej:
            i += 1
            lastrej = mask.sum()
            adj = np.subtract(data,np.median(data))
            mask = np.power(adj,2) <= np.median(data) * 9.
        np.savez('/Users/gray/Desktop/test.npz', data=data, mask=mask)
        return [data, mask]
    
    @cython.boundscheck(False)
    cdef int find_peak(self, str pn='pos'):
        cdef np.ndarray[double, ndim=1] data = np.asarray(<double[:self.size]> self.data)
        #cdef list c = self.clip()
        #if pn == 'pos':
        #    return np.nanargmax(npm.MaskedArray(c[0], ~c[1], fill_value=np.nan))
        #elif pn == 'neg':
        #    return np.nanargmax(npm.MaskedArray(c[0], ~c[1], fill_value=np.nan))
        return [np.nanargmax(data), np.nanargmin(data)][pn == 'neg']
    
    @cython.boundscheck(False)
    cdef CompositeModel fit_peaks(self, tuple p0, str model_type='Gaussian'):
        cdef:
            double median
            object p, pm, nm, fit
            list pos_models, neg_models, c
            np.ndarray[double, ndim=1] temp, data = np.asarray(<double[:self.size]> self.data)
            CompositeModel out = CompositeModel(p0, model_type=model_type)
        temp = replace_nans(data)
        self.data = <double*> temp.data
        c = self.clip()
        median = np.median(npm.MaskedArray(c[0], ~c[1]))
        fit = fit_to_model(out.model, np.arange(temp.size, dtype=np.float64), 
                           np.asarray(<double[:self.size]> self.data))
        self.data = <double*> data.data
        out = CompositeModel(grab_params(fit), model_type)
        return out

cdef class Extraction:
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, *args, **kw):
        if len(args) == 5:
            self.dither1 = args[0]
            self.dither2 = args[1]
            self.flat = args[2]
            self.on_axis = abs(args[4])
            self.off_axis = (self.on_axis + 1) % 2
            self.region = args[3]
        else:
            self.dither1 = np.empty([0,0], dtype=np.float64)
            self.dither2 = np.empty([0,0], dtype=np.float64)
            self.flat = np.empty([0,0], dtype=np.float64)
            self.off_axis = -1
            self.on_axis = -1
            self.region = []
        self.mtype = str(kw.get('mtype', 'Gaussian'))
        self.distorts = {'pos':[], 'neg':[]}
        self.extracts = {'spec': np.empty([0,0], dtype=np.float64), 
                         'tell': np.empty([0,0], dtype=np.float64), 
                         'lamp': np.empty([0,0], dtype=np.float64)}
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, *args, **kw):
        self.rebuild()
    
    property name:
        def __get__(self):
            return str(hash(self.file1+':'+self.file2))
    
    property diff:
        def __get__(self):
            return np.asarray(self.diff_full)
    
    property extract_region:
        def __get__(self):
            return np.asarray(self.diff0)
    
    cpdef rebuild(self):
        cdef:
            list region = self.region[:]
            np.ndarray[double, ndim=2] dithera = np.asarray(self.dither1)
            np.ndarray[double, ndim=2] ditherb = np.asarray(self.dither2)
            np.ndarray[double, ndim=2] flat = np.asarray(self.flat)
        region = [region[1], region[0], region[3], region[2]]
        self.diff_full = np.subtract(self.dither1, self.dither2)
        self.reg1 = np.divide(dithera[region[0]:region[2], region[1]:region[3]], 
                             flat[region[0]:region[2], region[1]:region[3]])
        self.reg2 = np.divide(ditherb[region[0]:region[2], region[1]:region[3]],
                              flat[region[0]:region[2], region[1]:region[3]])
        self.diff0 = np.subtract(self.reg1, self.reg2)
        self.tell0 = np.minimum(self.reg1, self.reg2)
    
    @cython.boundscheck(False)
    cdef CompositeModel fit_section(self, int cen, CompositeModel model, 
                                    int width, Py_ssize_t nx, Py_ssize_t ny):
        cdef:
            np.ndarray[Py_ssize_t, ndim=2] xr, yr
            Py_ssize_t[:] axis = np.arange(-width,width)
            OneDFitting section = OneDFitting()
        
        xr = [np.tile(np.clip(np.add(axis,cen), 0, nx-1)[..., np.newaxis], (1, ny)),
              np.tile(np.arange(nx)[..., np.newaxis], (1, axis.size))][self.on_axis]
        yr = [np.tile(np.arange(ny)[np.newaxis, ...], (axis.size, 1)),
              np.tile(np.clip(np.add(axis,cen), 0, ny-1)[np.newaxis, ...], (nx, 1))][self.on_axis]
        section.from_2d(np.asarray(self.diff0, dtype=np.float64)[xr, yr], axis=self.on_axis)
        return section.fit_peaks(grab_params(model.model), model.mtype)
    
    @cython.boundscheck(False)
    cpdef np.ndarray[double, ndim=3] fit_trace(self, list p0, 
                                               np.ndarray[double, ndim=2] lamp, 
                                               bint lamps=False, int width=10, 
                                               bint extract=False):
        cdef:
            CompositeModel model = CompositeModel(tuple(p0), model_type=self.mtype)
            OneDFitting profile = OneDFitting()
            int nx = self.diff0.shape[0], ny = self.diff0.shape[1]
            int n_wav = self.diff0.shape[self.on_axis], n_ap = len(p0)
            Py_ssize_t cen = n_wav/2 + (n_wav % 2), c
            CompositeModel fit_profile, up, down
            np.ndarray[double, ndim=3] all_parameters = np.empty((n_wav, n_ap, len(p0[0])), dtype=np.float64)
            np.ndarray[double, ndim=2] diff = np.asarray(self.diff0).transpose(self.on_axis, self.off_axis)
            np.ndarray[double, ndim=2] tell = np.asarray(self.tell0).transpose(self.on_axis, self.off_axis)
            np.ndarray[double, ndim=2] lamp_
            np.ndarray[double, ndim=1] stripe = np.arange(self.diff0.shape[self.off_axis], dtype=np.float64)
        

        #first we grab the center section
        fit_profile = self.fit_section(cen, model, width, nx, ny)
        all_parameters[cen] = np.array(grab_params(fit_profile.model), dtype=np.float64)
        up, down = fit_profile.copy(), fit_profile.copy()
        if extract:
            self.extracts['spec'] = np.empty((n_wav, n_ap), dtype=np.float64)
            self.extracts['tell'] = np.empty((n_wav, n_ap), dtype=np.float64)
            self.extracts['lamp'] = np.empty((n_wav, n_ap), dtype=np.float64)
            self.extracts['spec'][cen] = np.nansum(fit_profile.evaluate_all(stripe) * np.tile(diff[cen][np.newaxis, ...], (n_ap, 1)), axis=1)
            self.extracts['tell'][cen] = np.nansum(fit_profile.evaluate_all(stripe) * np.tile(tell[cen][np.newaxis, ...], (n_ap, 1)), axis=1)
            if lamps:
                lamp_ = lamp.transpose(self.on_axis, self.off_axis)
                self.extracts['lamp'][cen] = np.nansum(fit_profile.evaluate_all(stripe) * np.tile(lamp_[cen][...,np.newaxis], (1, n_ap)), axis=0)
        for c in xrange(1, cen+1): #now work outward from the middle
            if cen + c < [nx, ny][self.on_axis]:
                up = self.fit_section(cen + c, up, width, nx, ny)  
                all_parameters[cen+c] = np.array(grab_params(up.model), dtype=np.float64)
                if extract:
                    self.extracts['spec'][cen+c] = np.nansum(up.evaluate_all(stripe) * np.tile(diff[cen+c][np.newaxis, ...], (n_ap, 1)), axis=1)
                    self.extracts['tell'][cen+c] = np.nansum(up.evaluate_all(stripe) * np.tile(tell[cen+c][np.newaxis, ...], (n_ap, 1)), axis=1)
                    if lamps:
                        self.extracts['lamp'][cen+c] = np.nansum(up.evaluate_all(stripe) * np.tile(lamp_[cen+c][np.newaxis, ...], (n_ap, 1)), axis=1)
            if cen - c >= 0:
                down = self.fit_section(cen - c, down, width, nx, ny)
                all_parameters[cen-c] = np.array(grab_params(down.model), dtype=np.float64)
                if extract:
                    self.extracts['spec'][cen-c] = np.nansum(down.evaluate_all(stripe) * np.tile(diff[cen-c][np.newaxis, ...], (n_ap, 1)), axis=1)
                    self.extracts['tell'][cen-c] = np.nansum(down.evaluate_all(stripe) * np.tile(tell[cen-c][np.newaxis, ...], (n_ap, 1)), axis=1)
                    if lamps:
                        self.extracts['lamp'][cen-c] = np.nansum(down.evaluate_all(stripe) * np.tile(lamp_[cen-c][np.newaxis, ...], (n_ap, 1)), axis=1)
        return all_parameters
    
    @cython.boundscheck(False)
    cpdef fix_distortion(self, double[:, :, :] centers):
        '''Fit the path of the trace with a polynomial, and warp the image
        back to straight.'''
        cdef OneDFitting centers_pos = OneDFitting(), centers_neg = OneDFitting()
        cdef np.ndarray[double, ndim=1] distort_pos, distort_neg
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] pos, mask
        cdef np.ndarray[double, ndim=2] amp = np.asarray(centers, dtype=np.float64)[:,:,0]
        cdef np.ndarray[double, ndim=2] cent = np.asarray(centers, dtype=np.float64)[:,:,1]
        cdef Py_ssize_t mid = cent.shape[0]/2
        cdef object fit_model
        pos = np.greater(amp[mid, :], (amp[mid,:].max() + amp[mid,:].min())/2.)
        centers_pos.from_2d((cent - np.tile(cent[0,:][np.newaxis, ...], (cent.shape[0], 1)))[:,pos], axis=1)
        centers_neg.from_2d((cent - np.tile(cent[0,:][np.newaxis, ...], (cent.shape[0], 1)))[:,~pos], axis=1)
        distort_pos = np.asarray(<double[:centers.shape[0]]>centers_pos.data)
        distort_neg = np.asarray(<double[:centers.shape[0]]>centers_neg.data)
        fit_model = fit_to_model(models.Polynomial1D(degree=2), 
                                 np.arange(distort_pos.size, dtype=np.float64), 
                                 distort_pos)
        mask = ~sigma_clip(fit_model(np.arange(distort_pos.size, dtype=np.float64)) - distort_pos, 2).mask
        fit_model = fit_to_model(models.Polynomial1D(degree=2),
                                 np.arange(np.count_nonzero(mask), 
                                           dtype=np.float64), distort_pos[mask])
        #self.distorts['pos'] = distort_pos.tolist()
        self.distorts['pos'] = fit_model.parameters.tolist()
        fit_model = fit_to_model(models.Polynomial1D(degree=2), 
                                 np.arange(distort_neg.size, dtype=np.float64), 
                                 distort_neg)
        mask = ~sigma_clip(fit_model(np.arange(distort_neg.size, dtype=np.float64)) - distort_neg, 2).mask
        fit_model = fit_to_model(models.Polynomial1D(degree=2),
                                 np.arange(np.count_nonzero(mask), 
                                           dtype=np.float64), distort_neg[mask])
        #self.distorts['neg'] = distort_neg.tolist()
        self.distorts['neg'] = fit_model.parameters.tolist()
        self.undistort()
        self.rebuild()
    
    @cython.boundscheck(False)
    cdef void undistort(self):
        if self.on_axis:
            self.dither1 = geometric_transform(self.dither1.T, undistort_function(np.array(self.distorts['pos'], dtype=np.float64))).T
            self.dither2 = geometric_transform(self.dither2.T, undistort_function(np.array(self.distorts['neg'], dtype=np.float64))).T
        else:
            self.dither1 = geometric_transform(self.dither1, undistort_function(np.array(self.distorts['pos'], dtype=np.float64)))
            self.dither2 = geometric_transform(self.dither2, undistort_function(np.array(self.distorts['neg'], dtype=np.float64)))
    
    def __getstate__(self):
        return {'region':self.region, 'files':[self.file1, self.file2, self.flatfile],
                'axes':[self.on_axis, self.off_axis], 'mtype':self.mtype,
                'distortion':self.distorts}
    
    def __setstate__(self, state):
        cdef list header = [0]
        self.file1 = str(state['files'][0])
        self.file2 = str(state['files'][1])
        self.flatfile = str(state['files'][2])
        self.dither1 = fitsimage(self.file1, header)
        self.dither2 = fitsimage(self.file2, header)
        self.flat = fitsimage(self.flatfile, header)
        self.region = state['region']
        self.on_axis = state['axes'][0]
        self.off_axis = state['axes'][1]
        self.mtype = str(state['mtype'])
        self.distorts = state['distortion']
        if self.distorts['pos'] or self.distorts['neg']:
            self.undistort()
        self.rebuild()
    
def findpeaks1d(np.ndarray[double, ndim=1] data, str pn='pos'):
    cdef OneDFitting finder = OneDFitting()
    finder.from_1d(data)
    return finder.find_peak(pn)

def findpeaks2d(np.ndarray[double, ndim=2] data, int axis=0, str combine_type='median', str pn='pos'):
    cdef OneDFitting finder = OneDFitting()
    finder.from_2d(data, axis=axis, combine_type=combine_type)
    return finder.find_peak(pn)

def fitpeaks1d(np.ndarray[double, ndim=1] data, tuple parameters, str model_type='Gaussian'):
    cdef OneDFitting fitter = OneDFitting()
    fitter.from_1d(data)
    return fitter.fit_peaks(parameters, model_type=model_type)

def fitpeaks2d(np.ndarray[double, ndim=2] data, tuple parameters, str model_type='Gaussian'):
    cdef OneDFitting fitter = OneDFitting()
    fitter.from_2d(data)
    return fitter.fit_peaks(parameters, model_type=model_type)

def extraction(np.ndarray[double, ndim=2] dither_1, np.ndarray[double, ndim=2] dither_2,
               np.ndarray[double, ndim=2] flat, list region, int tracedir=-1, 
               str model_type='Gaussian'):
    cdef Extraction extractor = Extraction(dither_1, dither_2, flat, region, tracedir, mtype=model_type)
    return extractor

def extraction_from_state(dict state, np.ndarray[double, ndim=2] flat):
    cdef:
        Extraction extractor
        object hdu
        np.ndarray[double, ndim=2] dithera, ditherb
    hdu = fits.open(state['files'][0])
    dithera = hdu[0].data.astype(np.float64)
    hdu.close()
    hdu = fits.open(state['files'][1])
    ditherb = hdu[0].data.astype(np.float64)
    hdu.close()
    extractor = Extraction(dithera, ditherb, flat, state['region'], -1,
                           state['mtype'])
    extractor.on_axis = state['axes'][0]
    extractor.off_axis = state['axes'][1]
    extractor.file1 = state['files'][0]
    extractor.file2 = state['files'][1]
    extractor.distorts = state['distortion']
    extractor.undistort()
    extractor.rebuild()
    return extractor
