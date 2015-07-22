# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:42:10 2015

@author: gray
"""

import numpy as np
cimport numpy as np
cimport cython
 
import sys
from astropy.modeling import models, fitting
from scipy.misc import bytescale
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.interpolate import InterpolatedUnivariateSpline#, interp2d
from astropy.io import fits
import os, re, copy
from scipy.signal import medfilt#, medfilt2d

cdef bint little = sys.byteorder == 'little'

cdef class Spectrum:
    
    def __cinit__(self, np.ndarray[double, ndim=1] w,
                  np.ndarray[double, ndim=1] s):
        self._spec = s
        self._wav = w
    
    property spec:
        def __get__(self):
            return np.asarray(self._spec, dtype=np.float64)
    
        def __set__(self, np.ndarray[double, ndim=1] new):
            self._spec = new

    property wav:
        def __get__(self):
            return np.asarray(self._spec, dtype=np.float64)
        
        def __set__(self, np.ndarray[double, ndim=1] new):
            self._wav = new

cpdef np.ndarray[double, ndim=1] replace_nans(np.ndarray[double, ndim=1] data):
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] nans = ~np.isfinite(data)
    cdef np.ndarray[double, ndim=1] smoothed = medfilt(data, 3)
    cdef np.ndarray[double, ndim=1] out = data.copy()
    out[nans] = smoothed[nans]
    return out

cpdef object fit_to_model(object model, np.ndarray[double, ndim=1] independent,
                         np.ndarray[double, ndim=1] dependent):
    cdef object fit = fitting.LevMarLSQFitter()
    return fit(model, independent, dependent)

@cython.wraparound(False)
cpdef fitsimage(str filename, list header):
    cdef object hdu, image
    hdu = fits.open(filename)
    if little: #fits files are always big-endian
        image = hdu[0].data.byteswap().newbyteorder().astype(np.float64)
    else:
        image = hdu[0].data.astype(np.float64)
    header[0] = hdu[0].header
    hdu.close()
    return image

cpdef np.ndarray[double, ndim=1] twod_to_oned(np.ndarray[double, ndim=2] array,
                                             int axis=0, str method='mean'):
    return np.median(array, axis=axis)

cpdef np.ndarray[double, ndim=1] interp(np.ndarray[double, ndim=1] dependent, 
                                       np.ndarray[double, ndim=1] independent,
                                       np.ndarray[double, ndim=1] target): 
    return InterpolatedUnivariateSpline(independent, dependent, ext=0,
                                        w=np.isfinite(dependent))(target)

cdef class ScaleableStack:
    
    @cython.boundscheck(False)
    cdef void _scale(self):
        cdef:
            Py_ssize_t i
            object p_init = models.Linear1D(slope=1, intercept=0), p
            np.ndarray[double, ndim=1] layer
        if self.scaled: return
        for i, layer in enumerate(self._stack[1:]):
            #linear fit to find scale factors
            p_init = models.Linear1D(slope=1, intercept=0)
            p = fit_to_model(p_init, layer, self._stack[0])
            self._stack[i] = p(layer)
        self.scaled = True
    
    def __getitem__(self, int index):
        return self._stack[index]
    
    @cython.boundscheck(False)
    def __len__(self):
        return len(self._stack)

cdef class ImageStack(ScaleableStack):
    
    @cython.boundscheck(False)
    def __cinit__(self, *args, **kw):
        self.scaled = False
        self._stack = []
        self.stack_list = []
        self.headers = []
        self.dithers = []
        self.ditherpairs = []
        self.files_card = ''
        
    def __init__(self, str filestring, str stub):
        cdef: 
            str spotlen, base, reg = '#+', t, x, ff
            np.ndarray[double, ndim=2] image
            object spot
            list files, tmp, f, a, b, aa, bb
            float na, nb
            Py_ssize_t i, j
            int _
        
        if filestring == '' or filestring == 'null':
            return
        header = [0]
        self.filestring = filestring
        self.stub = stub
        if len(re.findall(reg, stub)) != 1:
            raise ValueError("File format is not valid; must use '#' as placeholder only")
        spot = re.search(reg, stub)
        spotlen = str(spot.end() - spot.start())
        base = re.sub(reg, '%0'+spotlen+'d', stub)
        files = []
        tmp = re.split('[.,]', filestring)
        for t in tmp:
            f = re.split('-', t)
            if len(f) == 1:
                files.append(int(f))
            else:
                for i in range(len(f) - 1):
                    for j in range(int(f[i]), int(f[i+1])+1):
                        files.append(j)
        self.stack_list = [(base + '.fits') % _ for _ in files]
        self.files_card = filestring
        self.dithers = ['ABBA'[i % 4] for i in range(len(files))]
        for ff in self.stack_list:
            image = fitsimage(ff, header)
            self._stack.append(image)
            self.headers.extend(header)
        a = [i for i, x in enumerate(self.dithers) if x == 'A']
        b = [i for i, x in enumerate(self.dithers) if x == 'B']
        na, nb = float(len(a)), float(len(b))
        aa = [a[i] for i in (np.arange(nb)*na/nb).astype('int')] if na > nb else a
        bb = [b[i] for i in (np.arange(na)*na/nb).astype('int')] if nb > na else b
        self.ditherpairs = zip(aa, bb)
    
    property combined:
        def __get__(self):
            return np.asarray(self._combine)
    
    def __getstate__(self):
        return {'filestring': self.filestring, 'stub': self.stub}
    
    def __setstate__(self, kw):
        self.__init__(str(kw.get('filestring', 'null')), str(kw.get('stub', '')))
    
    @cython.boundscheck(False)
    cpdef list medcombine(self, str outputfile=''):
        cdef object outfits
        if not self.scaled: self.scale()
        self._combine = np.median(self._stack, axis=0)
        finalheader = self.headers[0]
        finalheader['FILES'] = self.files_card
        if outputfile:
            outfits = fits.HDUList([fits.PrimaryHDU(data=self._combine, header=finalheader)])
            outfits.verify('fix')
            outfits.writeto(outputfile, output_verify='fix', clobber=True)
        return [finalheader, self._combine]
    
    @cython.boundscheck(False)
    cpdef scale(self):
        cdef: 
            np.ndarray[double, ndim=2] layer
            np.ndarray[double, ndim=1] flat
            list shapes = []
            Py_ssize_t i
        for i, layer in enumerate(self._stack):
            shapes.append((layer.shape[0], layer.shape[1]))
            self._stack[i] = layer.flatten()
        self._scale()
        for i, flat in enumerate(self._stack):
            self._stack[i] = flat.reshape(shapes[i])

cdef class SpectrumStack(ScaleableStack):
    
    def __cinit__(self, *args, **kw):
        self.scaled = False
        self.headers = []
        self._stack = []
        self.stack_list = []
    
    def __init__(self, list spectrum_list):
        cdef ExtractedSpectrum s
        for s in spectrum_list:
            self._stack.append(Spectrum(s.wav, s.spec))
            self.stack_list.append(s.filename)
            self.headers.append(s.header)
    
    property combined:
        def __get__(self):
            return Spectrum(self._x, self._combine)
    
    cpdef scale(self, Py_ssize_t index=0):
        cdef: 
            np.ndarray[double, ndim=1] flux, ref_x = self._stack[index].wav
            Spectrum spec
        self._stack = [interp(replace_nans(spec.spec), spec.wav, ref_x) for spec in self._stack]
        self._scale()
        self._stack = [Spectrum(ref_x, flux) for flux in self._stack]
    
    @cython.boundscheck(False)
    cpdef Spectrum combine(self, bint median=True): 
        cdef:
            np.ndarray[double, ndim=1] first_pass, smooth, tmp
            np.ndarray[double, ndim=2] weighted
            Spectrum spec
            list weights, stack = []
            double tot
            Py_ssize_t i
        if not self.scaled:
            self.scale()
        stack = [replace_nans(spec.spec) for spec in self._stack]
        #stack = [spec.spec for spec in self._stack]
        self._x = self._stack[0].wav
        first_pass = np.median(stack, axis=0)
        if median:
            return Spectrum(np.asarray(self._x), first_pass)
        smooth = medfilt(first_pass, kernel_size=5)
        weights = []
        for spec in stack:
            tmp = np.power(spec.spec - smooth, 2)
            tmp = np.clip(tmp, min=0.5, max=tmp.max())
            weights.append(np.reciprocal(tmp))
        tot = np.array(weights).sum(axis=0)
        weighted = np.array([spec.spec * weights[i] / tot for i, spec in enumerate(stack)])
        self._combine = weighted.sum(axis=0)
        return Spectrum(self._x, self.combined)
    
    @cython.boundscheck(False)
    cpdef SpectrumStack subset(self, list quality):
        cdef:
            SpectrumStack out
            Spectrum x
            object h
            Py_ssize_t i

        out = copy.deepcopy(self)
        out._stack = [x for i,x in enumerate(self._stack) if quality[i]]
        if out.headers:
            out.headers = [h for i,h in enumerate(out.headers) if quality[i]]
        return out

cdef class ScalableImage:

    @cython.boundscheck(False)
    cpdef load(self, np.ndarray[double, ndim=2] data, str scalemode='linear', 
               double factor=0.):
        cdef list threshold
        
        self._data = data
        self.nx, self.ny = data.shape[1], data.shape[0]
        threshold = list(self.zscale())
        self.lo, self.hi = threshold[0], threshold[1]
        self.factor = factor
        self.mode = scalemode
        self.imstretch()
    
    @cython.boundscheck(False)
    cdef zscale(self, float contrast=0.25, int num_points=600, int num_per_row=120):
        cdef:
            int num_per_col, xsize, ysize, center_pixel
            float row_skip, col_skip
            np.ndarray[double, ndim=1] data, x_data
            Py_ssize_t i, j, x, y
            double data_min, data_max, med, z1, z2, zmin, zmax
            object p_init = models.Linear1D(1, 0), p
        
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self._data.shape[0], self._data.shape[1]
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = np.empty(num_per_row * num_per_col)
        for i in xrange(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in xrange(num_per_col):
                y = int(j * col_skip + 0.5)
                data[j+num_per_col*i] = self._data[x, y]
        data.sort()
        data = replace_nans(data)
        data_min, data_max = data.min(), data.max()
        center_pixel = (num_points + 1) / 2
        if data_min == data_max:
            return [data_min, data_max]
        med = np.median(data)
        data = sigma_clip(data, 3.).compressed()
        if data.size < int(num_points/2.0):
            return data_min, data_max
        x_data = np.arange(data.size, dtype=np.float64)
        #p_init = models.Linear1D(1, 0)
        p = fit_to_model(p_init, x_data, data)
        z1 = med - (center_pixel-1) * p.slope / contrast
        z2 = med + (num_points - center_pixel) * p.slope / contrast
        zmin = max(z1, data_min)
        zmax = min(z2, data_max)
        if zmin >= zmax:
            return data_min, data_max
        return zmin, zmax
    
    property data:
        def __get__(self):
            return self._data
    
    property scaled:
        def __get__(self):
            return self._scaled
    
    property dimensions:
        def __get__(self):
            return [self.nx, self.ny]
        
    property threshold:
        def __get__(self):
            return [self.lo, self.hi]
    
    @cython.boundscheck(False)
    cpdef change_parameters(self, dict info):
        self.nx = info.get('min', self.nx)
        self.ny = info.get('max', self.ny)
        self.mode = info.get('mode', self.mode)
        self.factor = info.get('factor', self.factor)
        self.imstretch()
    
    @cython.boundscheck(False)
    cdef void imstretch(self):
        cdef:
            np.ndarray[double, ndim=2] data = np.clip(self._data, self.lo, self.hi)
            double mn, mx
            np.ndarray[double, ndim=1] beta, sclbeta, nonlinearity, extrema, bins
            np.ndarray[int, ndim=1] imhist, cdf
        if self.mode == 'linear':
            pass
        elif self.mode == 'logarithmic':
            data = np.reciprocal(1 + np.power(0.5 / data, self.factor))
        elif self.mode == 'gamma':
            data = np.power(data, self.factor)
        elif self.mode == 'arcsinh':
            mn = np.nanmin(data)
            mx = np.nanmax(data)
            beta = np.clip(self.factor, 0., self.factor)
            sclbeta = (beta - mn) / (mx - mn)
            sclbeta = np.clip(sclbeta, 1.e-12, sclbeta)
            nonlinearity = 1. / sclbeta
            extrema = np.arcsinh(np.array([0., nonlinearity]))
            data = np.clip(np.arcsinh(data * nonlinearity), extrema[0], extrema[1])
        elif self.mode == 'square root':
            data = np.sqrt(np.fabs(data))*np.sign(data)
        elif self.mode == 'histogram equalization':
            imhist, bins = np.histogram(data.flatten(),256,normed=True)
            cdf = imhist.cumsum() #cumulative distribution function
            cdf = 255 * cdf / cdf[-1] #normalize
            data = np.interp(data.flatten(),bins[:-1],cdf).reshape(data.shape[0], data.shape[1])
        self._scaled = bytescale(data).flatten().tolist()

@cython.boundscheck(False)
@cython.wraparound(False)
def scalable_image(np.ndarray[double, ndim=2] data):
    cdef ScalableImage out = ScalableImage()
    out.load(data)
    return out

cdef class InstrumentProfile:
    
    def __cinit__(self, **kw):
        self.instid = str(kw.get('instid', ''))
        self.tracedir = str(kw.get('tracedir', 'horizontal'))
        self.dimensions = list(kw.get('dimensions', [1024, 1024]))
        self.headerkeys = kw.get('headerkeys', {'exp':'EXPTIME', 
                                                'air':'AIRMASS', 
                                                'type':'IMAGETYP'})
        self.description = str(kw.get('description', ''))
    
    def __getstate__(self):
        return {'instid': self.instid,
                'tracedir': self.tracedir,
                'dimensions': self.dimensions,
                'headerkeys': self.headerkeys,
                'description': self.description}
    
    def __setstate__(self, state):
        self.instid = str(state.get('instid', ''))
        self.tracedir = str(state.get('tracedir', 'horizontal'))
        self.dimensions = list(state.get('dimensions', [1024, 1024]))
        self.headerkeys = state.get('headerkeys', {'exp':'EXPTIME', 
                                                   'air':'AIRMASS', 
                                                   'type':'IMAGETYP'})
        self.description = str(state.get('description', ''))
        
cdef class ObsTarget:
    
    def __cinit__(self, *args, **kw):
        self.targid = str(kw.get('targid', ''))
        self.instrument_id = str(kw.get('instrument_id', ''))
        self.filestring = str(kw.get('filestring', ''))
        self.notes = str(kw.get('notes', ''))
        self.extractions = kw.get('extractions', {})
        self.dither = kw.get('dither', [])
        self.spectra = kw.get('spectra', [])
        self.ditherpairs = kw.get('ditherpairs', [])
    
    def __init__(self, *args, **kw):
        self.images = kw.get('images', ImageStack('null',''))
    
    def __getstate__(self):
        cdef str key
        return dict(zip(['targid', 'instrument_id', 'filestring', 'notes',
                         'extractions', 'dither', 'spectra', 'ditherpairs',
                         'images'], [self.targid, self.instrument_id,
                         self.filestring, self.notes, self.extractions,
                         self.dither, self.spectra, self.ditherpairs, self.images]))

    def __setstate__(self, state):
        self.targid = str(state.get('targid', ''))
        self.instrument_id = str(state.get('instrument_id', ''))
        self.filestring = str(state.get('filestring', ''))
        self.notes = str(state.get('notes', ''))
        self.extractions = state.get('extractions', {})
        self.dither = state.get('dither', [])
        self.spectra = state.get('spectra', [])
        self.ditherpairs = state.get('ditherpairs', [])
        self.images = state.get('images', ImageStack('null',''))
    
    def __str__(self):
        return "Target {}, extractions: ".format(self.targid, {k:v.__getstate__() for k,v in self.extractions})

cdef class ObsNight:

    def __cinit__(self, **kw):
        self.date = str(kw.get('date', ''))
        self.filestub = str(kw.get('filestub', ''))
        self.rawpath = str(kw.get('rawpath', ''))
        self.outpath = str(kw.get('outpath', ''))
        self.calpath = str(kw.get('calpath', ''))
        self.targets = kw.get('targets', {})
        self.flaton = kw.get('flaton', [])
        self.flatoff = kw.get('flatoff', [])
        self.cals = kw.get('cals', [])
    
    cpdef bint add_to(self, ObsTarget element):
    #cpdef bint add_to(self, object element):
        #if not isinstance(element, ObsTarget):
        #    element = ObsTarget(**element)
        element.images = ImageStack(element.filestring, os.path.join(self.rawpath, self.filestub))
        element.dither = element.images.dither
        element.ditherpairs = element.images.ditherpairs
        self.targets[element.targid] = element
        return True

    cpdef ObsTarget get_from(self, str index):
        return self.targets.get(index, None)
    
    def __getstate__(self):
        return {'date': self.date, 'filestub': self.filestub, 
                'rawpath': self.rawpath, 'outpath': self.outpath,
                'calpath': self.calpath, 'targets': self.targets,
                'flaton': self.flaton, 'cals': self.cals, 
                'flatoff': self.flatoff}
    
    def __setstate__(self, kw):
        self.date = str(kw.get('date', ''))
        self.filestub = str(kw.get('filestub', ''))
        self.rawpath = str(kw.get('rawpath', ''))
        self.outpath = str(kw.get('outpath', ''))
        self.calpath = str(kw.get('calpath', ''))
        self.targets = kw.get('targets', {})
        self.flaton = kw.get('flaton', [])
        self.flatoff = kw.get('flatoff', [])
        self.cals = kw.get('cals', [])

    def __str__(self):
        return "Night {}, targets: {}".format(self.date, {k:str(v) for k,v in self.targets.items()})

cdef class ObsRun:
        
    def __cinit__(self, **kw):
        self.runid = str(kw.get('runid', ''))
        self.nights = kw.get('nights', {})
    
    cpdef bint add_to(self, ObsNight element):
    #cpdef bint add_to(self, object element):
        #if isinstance(element, ObsNight):
        #    self.nights[element.date] = element
        #else:
        #    self.nights[element['date']] = ObsNight(**element)
        self.nights[element.date] = element
        return True

    cpdef ObsNight get_from(self, str index):
        return self.nights.get(index, None)
    
    def __getstate__(self):
        return {'runid': self.runid,
                'nights': self.nights}
    
    def __setstate__(self, kw):
        self.runid = str(kw.get('runid', ''))
        self.nights = kw.get('nights', {})
    
    def __str__(self):
        return "ObsRun {}, nights: {}".format(self.runid, {k:str(v) for k,v in self.nights.items()})

cdef class ExtractedSpectrum:

    def __cinit__(self, str specfile):
        self.filename = ''
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, str specfile):
        cdef np.ndarray the_spec
        cdef list head = [0]
        self.filename = specfile
        the_spec = fitsimage(self.filename, head)
        self.header = head[0]
        if the_spec.ndim == 2:
            self._wav = the_spec[0,:]
            self._spec = the_spec[1,:]
        else:
            self._spec = the_spec
            self._wav = np.arange(self.spec.size, dtype=np.float64)
        
    property spec:
        def __get__(self):
            return np.asarray(self._spec, dtype=np.float64)
        def __set__(self, np.ndarray[double, ndim=1] data):
            self._spec = data
    
    property wav:
        def __get__(self):
            return np.asarray(self._wav, dtype=np.float64)
        def __set__(self, np.ndarray[double, ndim=1] data):
            self._wav = data
    
    property comb:
        def __get__(self):
            return np.vstack((self._wav,self._spec))
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef update_fits(self):
        cdef np.ndarray[double, ndim=2] data = self.comb
        fits.update(self.filename, data, self.header)    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef save(self, str outputfile=''):
        cdef np.ndarray[double, ndim=2] data = self.comb
        if outputfile == self.filename or outputfile == '':
            self.update_fits()
        else:
            fits.writeto(outputfile, data, header=self.header)