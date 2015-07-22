# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:12:49 2015

@author: gray
"""

import numpy as np
#from datatypes import FitsImage
from astropy.io import fits
#import sys

#def fitsimage(filename, header=None):
#    hdu = fits.open(filename)
#    if sys.byteorder == 'little': #fits files are always big-endian
#        image = hdu[0].data.byteswap().newbyteorder().astype(np.float64)
#    else:
#        image = hdu[0].data.astype(np.float64)
#    header = hdu[0].header
#    hdu.close()
#    return image


def imheader(func):
    def preserve_header(im1, im2, outputfile=None, *args, **kw):
        finalheader = None
        if hasattr(im1, 'header') and im1.header:
            finalheader = im1.header
        #_im1 = im1.data_array if isinstance(im1, FitsImage) else im1
        #_im2 = im2.data_array if isinstance(im2, FitsImage) else im2
        _im1, _im2 = im1, im2
        #outputimg = np.array(Robust2D(func(_im1, _im2, *args, **kw)).replace_nans())
        outputimg = np.ma.masked_invalid(func(_im1, _im2, *args, **kw)).filled(0.)
        if outputfile:
            outfits = fits.HDUList([fits.PrimaryHDU(data=outputimg, header=finalheader)])
            outfits.verify('fix')
            outfits.writeto(outputfile, output_verify='fix', clobber=True)
        #if isinstance(im1, FitsImage):
        #    tmp = FitsImage('', header=finalheader)
        #    super(FitsImage, tmp).load(outputimg)
        #    return tmp
        return outputimg
    preserve_header.__name__ = func.__name__
    return preserve_header

im_add = imheader(np.add)
im_divide = imheader(np.divide)
im_minimum = imheader(np.fmin)
im_subtract = imheader(np.subtract)
