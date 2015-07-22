# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:51:03 2015

@author: gray
"""


import numpy as np
#from scipy.misc import bytescale
from astropy.modeling import models, fitting 
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.interpolate import interp1d, interp2d
from astropy.io import fits
import os, re, copy
from collections import namedtuple, Mapping
from scipy.signal import medfilt, medfilt2d
from scalableimage import ScalableImage

placeholder = '#'
reg = placeholder+'+'

def parse_filestring(filestring, stub, preserve=True):
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
    images = ImageStack([(base + '.fits') % x for x in files], dithers=True, 
                         files_card=filestring, preserve=preserve)
    return images

def namedtuple_with_defaults(typename, field_names, default_values=[]):
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

InstrumentProfile = namedtuple_with_defaults('InstrumentProfile',['instid', 
    'tracedir', 'dimensions', 'headerkeys', 'description'], ['', 'horizontal', 
    (1024,1024), {'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, ''])

class ObsRun(namedtuple_with_defaults('ObsRun', ['runid', 'nights'], ['',{}])):
    def add_to(self, element):
        if isinstance(element, ObsNight):
            self.nights[element.date] = element
        else:
            self.nights[element['date']] = ObsNight(**element)
        return True
    
    def get_from(self, index):
        return self.nights.get(index, None)

class ObsNight(namedtuple_with_defaults('ObsNight', ['date','targets',
                                                     'filestub','rawpath',
                                                     'outpath','calpath',
                                                     'flaton','flatoff',
                                                     'cals'], ['',{},'','','',
                                                     '',[],[],[]])):
    def add_to(self, element):
        if not isinstance(element, ObsTarget):
            element = ObsTarget(**element)
        element.images, element.dither = parse_filestring(element.filestring, \
                os.path.join(self.rawpath, self.filestub))
        self.targets[element.targid] = element
        return True

    def get_from(self, index):
        return self.targets.get(index, None)

class ExtractedSpectrum(object):
    def __init__(self, specfile, data=False):
        if data:
            self.spec = specfile
            self.header = None
            self.data_input = True
            self.file = raw_input('Spectrum file name> ')
        else:
            self.file = specfile
            hdul = fits.open(specfile)
            hdu = hdul[0]
            self.header = hdu.header
            self.spec = hdu.data
            self.data_input = False
            hdu.close()
        if len(self.spec.shape) == 2:
            self.wav = self.spec[0,:]
            self.spec = self.spec[1,:]
        else:
            self.wav = np.arange(self.spec.size)
            
        
    def update_fits(self):
        data = np.vstack((self.wav,self.spec)) if self.wav else self.spec
        fits.update(self.file, data, self.header)    
    
    def save(self):
        if not self.data_input: self.update_fits()
        data = np.vstack((self.wav,self.spec)) if self.wav.size > 0 else self.spec
        fits.writeto(self.file, data, header=self.header)

class RobustData(np.ndarray):
    '''a wrapper to handle sigma-clipped statistics and interpolation.'''

    def __new__(cls, data, x=None, sigma=3., index=False):
        '''constructor as per numpy subclassing instructions'''
        obj = np.asarray(data).view(cls)
        if index or x==None:
            obj.x = obj.index
        elif x:
            obj.x = np.asarray(x)
        else:
            obj.x = None
        obj.sigma = sigma
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.sigma = getattr(obj, 'sigma', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def clipped(self, compress=False, inplace=False):
        clip = sigma_clip(self, self.sigma)
        if compress:
            clip = clip.compress()
        if not inplace:
            return clip
        self = clip
    
    @property
    def index(self):
        return np.arange(self.size)

    def stats(self, **kw):
        return sigma_clipped_stats(self, **kw)

    @property
    def normalized(self):
        return (self - np.median(self))

    @property
    def minmax(self):
        return np.sort(self)[[0,-1]]

    def fit_to_model(self, model, x=None, index=False):
        data = self.clipped()
        f = fitting.LevMarLSQFitter()
        if x is None:
            if index:
                x = self.index
            elif self.x is None:
                raise ValueError('Independent variable required.')
            else:
                x = self.x
        return f(model, x, data)

    def replace_nans(self, inplace=False):
        nans = np.isnan(self)
        tmp = medfilt(self, 3)
        if inplace:
            self[nans] = tmp[nans]
        else:
            y = self.copy()
            y[nans] = tmp[nans]
            return y

    def interp(self, target):
        f = interp1d(self.x, self.replace_nans(), kind='cubic')
        return f(target)

class Robust2D(RobustData):
    
    def interp(self, target_x, target_y):
        nx, ny = self.shape
        x, y = np.mgrid[0:nx,0:ny]
        f = interp2d(x, y, self.replace_nans(), kind='cubic')
        return f(target_x, target_y)
        
    def replace_nans(self, inplace=False):
        nans = np.isnan(self)
        tmp = medfilt2d(self, 3)
        if inplace:
            self[nans] = tmp[nans]
        else:
            y = self.copy()
            y[nans] = tmp[nans]
            return y
    
    def fit_to_model(self, model, xy=None):
        data = self.clipped()
        f = fitting.LevMarLSQFitter()
        if xy is None: 
            if self.x is None:
                raise ValueError('Independent variables required.')
            xy = self.xy
        x, y = xy
        return f(model, x, y, data)
    
    def combine(self, axis=0, method='mean', **kw):
        data = self.clipped()
        return {'mean':RobustData(np.mean(data, axis=axis)), 
                'median': RobustData(np.median(data, axis=axis))}[method]


def grab_header(file):
    hdul = fits.open(file)
    header = hdul[0].header
    hdul.close()
    return header

def zscale(imarray, contrast = 0.25, num_points = 600, num_per_row = 120):
    num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
    xsize, ysize = imarray.shape
    row_skip = float(xsize - 1) / float(num_per_row - 1)
    col_skip = float(ysize - 1) / float(num_per_col - 1)
    data = []
    for i in xrange(num_per_row):
        x = int(i * row_skip + 0.5)
        for j in xrange(num_per_col):
            y = int(j * col_skip + 0.5)
            data.append(imarray[x, y])
    data = RobustData(data)
    data.sort()
    data.replace_nans()
    data_min, data_max = data.min(), data.max()
    center_pixel = (num_points + 1) / 2
    if data_min == data_max:
        return data_min, data_max
    med = np.median(data)
    data.clipped(inplace=True)
    if data.size < int(num_points/2.0):
        return data_min, data_max
    x_data = np.arange(data.size)
    p_init = models.Linear1D(1, 0)
    p = data.fit_to_model(p_init, x=x_data)
    z1 = med - (center_pixel-1) * p.slope / contrast
    z2 = med + (num_points - center_pixel) * p.slope / contrast
    zmin = max(z1, data_min)
    zmax = min(z2, data_max)
    if zmin >= zmax:
        return data_min, data_max
    return zmin, zmax

'''class ScalableImage(object):
    def __init__(self):
        self.data_array = np.array([])
        self.dimensions = [0, 0]
        self.threshold = [0, 0]
        self.factor = 0.
        self.mode = ''

    def load(self, data, scalemode = 'linear', factor = None):
        self.data_array = data
        self.dimensions[1], self.dimensions[0] = self.data_array.shape
        self.threshold = zscale(self.data_array)
        self.factor = factor
        self.mode = scalemode
        self.imstretch()
    
    @property
    def data(self):
        return self.data_array
    
    def change_parameters(self, info):
        self.threshold = [info.get('min', self.threshold[0]), info.get('max', self.threshold[1])]
        self.mode = info.get('mode', self.mode)
        self.factor = info.get('factor', self.factor)
        self.imstretch()
    
    def imstretch(self):
        data = np.clip(self.data_array, self.threshold[0], self.threshold[1])
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
            im2 = np.interp(data.flatten(),bins[:-1],cdf)
            data = im2.reshape(data.shape)
        self.scaled = bytescale(data).flatten().tolist()'''

class FitsImage(ScalableImage):
    def __init__(self, fitsfile, header = None, load = False):
        super(FitsImage, self).__init__()
        self.fitsfile = fitsfile
        self.header = header or grab_header(fitsfile)
        if load:
            self.load()
    
    def load(self, **kwargs):
        hdu = fits.open(self.fitsfile)
        super(FitsImage, self).load(hdu[0].data.astype(float), **kwargs)
        hdu.close()
    
    def update_fits(self, header_only = False):
        if header_only:
            hdu = fits.open(self.fitsfile, mode='update')
            hdu[0].header = self.header
            hdu.flush()
            hdu.close()
            return
        fits.update(self.fitsfile, self.data_array, self.header)
    
    def __getstate__(self):
        return self.fitsfile, self.header, self.data_array.size > 0
    
    def __setstate__(self, state):
        self.__init__(state[0], header=state[1], load=state[2])
    
    def get_header_keyword(self, *args):
        if len(args) > 1:
            return [self.header.get(x, None) for x in args]
        else:
            return self.header.get(args[0], None)

fit = fitting.LevMarLSQFitter()

class ScaleableStack(object):
    def scale(self):
        if self.scaled: return
        for i, layer in enumerate(self.stack):
            #linear fit to find scale factors
            p_init = models.Linear1D(slope=1, intercept=0)
            p = fit(p_init, layer.flatten(), self.stack[0].flatten())
            self.stack[i] = p(layer).reshape(layer.shape)
        self.scaled = True
    
    def __getitem__(self, index):
        return self.stack[index]
    
    def __len__(self):
        return len(self.stack)

class ImageStack(ScaleableStack):
    def __init__(self, image_list, files_card='', data=False, headers=None, 
                 dithers=False, preserve=False):
        if data:
            self.stack = image_list
            self.stack_list = None
        else:
            self.stack_list = image_list
            self.stack = []
            self.headers = []
            if preserve: self.images = []
            for ff in image_list:
                im = FitsImage(ff, load=True)
                self.stack.append(im.data)
                self.headers.append(im.header)
                if preserve: self.images.append(im)
        self.files_card = files_card
        self.dithers = []
        self.scaled = False
        if dithers: 
            self.dithers = ['ABBA'[x % 4] for x in range(len(image_list))]
            a = [i for i, x in enumerate(self.dithers) if x == 'A']
            b = [i for i, x in enumerate(self.dithers) if x == 'B']
            na, nb = float(len(a)), float(len(b))
            aa = [a[i] for i in (np.arange(nb)*na/nb).astype('int')] if na > nb else a
            bb = [b[i] for i in (np.arange(na)*na/nb).astype('int')] if nb > na else b
            self.ditherpairs = zip(aa, bb)
    
    def __setstate__(self, args):
        args = list(args)
        scaled = args.pop()
        self.__init__(*args)
        if scaled:
            self.scale()
    
    def __getstate__(self):
        return (self.stack_list, self.files_card, self.stack_list==None, 
                self.headers, len(self.dithers) > 0, hasattr(self, 'images'), 
                self.scaled)
    
    #def next(self):
    #    nxt = self.images[self.itercount]
    #    self.itercount += 1
    #    return nxt
        
    #def __iter__(self):
    #    for im in self.images:
    #        yield im
        
    def medcombine(self, outputfile=None):
        if not self.scaled: self.scale()
        self.med_image = np.median(self.stack, axis=0)
        finalheader = self.headers[0]
        finalheader['FILES'] = self.files_card
        if outputfile:
            outfits = fits.HDUList([fits.PrimaryHDU(data=self.med_image, header=finalheader)])
            outfits.verify('fix')
            outfits.writeto(outputfile, output_verify='fix', clobber=True)
        return finalheader, self.med_image

class SpectrumStack(ScaleableStack):
    def __init__(self, spectrum_list, data=False, headers=None):
        self.scaled = False
        if data:
            self.stack = [RobustData(spec[1], x=spec[0]) for spec in spectrum_list]
            self.headers = headers
        else:
            self.stack = []
            self.headers = []
            for ff in spectrum_list:
                hdu = fits.open(ff)
                self.headers.append(hdu[0].header)
                data = hdu[0].data
                x = data[0]
                y = data[1]
                self.stack.append(RobustData(y, x=x))
                hdu.close()
    
    def __setstate__(self, args):
        scaled = args.pop()
        self.__init__(*args)
        if scaled:
            self.scale()
    
    def __getstate__(self):
        return (self.stack_list, self.files_card, self.stack_list==None, 
                self.headers, self.scaled)
    
    def scale(self, index=0):
        ref_x = self.stack[index].x
        self.stack = [RobustData(spectrum.interp(ref_x), x=ref_x) for spectrum in self.stack]
        super(SpectrumStack, self).scale()
    
    def combine(self, median=True):    
        first_pass = np.median(self.stack, axis=0)
        if median:
            return first_pass
        smooth = medfilt(first_pass, kernel_size=5)
        weights = []
        for spectrum in self.stack:
            tmp = np.power(spectrum - smooth, 2)
            tmp = np.clip(tmp, min=0.5, max=tmp.max())
            weights.append(np.reciprocal(tmp))
        tot = np.array(weights).sum(axis=0)
        weighted = np.array([s * weights[i] / tot for i, s in enumerate(self.stack)])
        self.spec_combine = RobustData(weighted.sum(axis=0))
        self.spec_combine.x = self.stack[0].x
        return self.spec_combine
    
    def subset(self, quality):
        tmp = copy.deepcopy(self)
        tmp.stack = [x for i,x in enumerate(self.stack) if quality[i]]
        if tmp.headers:
            tmp.headers = [x for i,x in enumerate(self.stack) if quality[i]]
        return tmp

ObsTarget = namedtuple_with_defaults('ObsTarget',['targid', 'instrument_id', 'filestring',
    'notes', 'images', 'extractions', 'dither', 'spectra', 'ditherpairs'], 
    ['','','','',ImageStack([]),{},[],[],[]])

if __name__ == '__main__':
    pass
