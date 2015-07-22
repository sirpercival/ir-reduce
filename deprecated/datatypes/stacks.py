# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:34:35 2015

@author: gray
"""

from astropy.io import fits
import numpy as np
from astropy.modeling import fitting, models
from robustdata import RobustData
from scipy.signal import medfilt
from fitsimage import FitsImage

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

class ImageStack(ScaleableStack):
    def __init__(self, image_list, data=False, headers=None, dithers=False):
        if data:
            self.stack = image_list
            self.stack_list = None
        else:
            self.stack_list = image_list
            self.stack = []
            self.headers = []
            for ff in image_list:
                im = FitsImage(ff, load=True)
                self.stack.append(im.data_array)
                self.headers.append(im.header)
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
        
        
    def median_combine(self, outputfile=None):
        if not self.scaled: self.scale()
        self.med_image = np.median(self.stack, axis=0)
        finalheader = self.headers[0]
        finalheader['FILES'] = self.stack_list
        if outputfile:
            outfits = fits.HDUList([fits.PrimaryHDU(data=self.med_image, header=finalheader)])
            outfits.verify('fix')
            outfits.writeto(outputfile, output_verify='fix', clobber=True)
        return finalheader, self.med_image

class SpectrumStack(ScaleableStack):
    def __init__(self, spectrum_list, data=False, headers=None):
        if data:
            self.stack = spectrum_list
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

    def scale(self):
        ref_x = self.stack[0].x
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
        weighted = [s * weights[i] / tot for i, s in enumerate(self.stack)]
        self.spec_combine = weighted.sum(axis=0)
        return self.spec_combine
