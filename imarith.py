from astropy.io import fits
from astropy.modeling import functional_models as fm, fitting
from numpy import *
from scipy.constants import golden
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from colorsys import hsv_to_rgb
from random import random

def grab_image_stack(imlist):
    images = []
    headers = []
    for ff in imlist:
        hdu = fits.open(ff)
        headers.append(hdu[0].header)
        images.append(hdu[0].data)
        hdu.close()
    return headers, images
    
def im_add(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = im1.data_array + im2.data_array
    write_fits(outputfile, finalheader, outputimg)
    return finalheader, outputimg

def im_divide(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = im1.data_array / im2.data_array
    write_fits(outputfile, finalheader, outputimg)
    return finalheader, outputimg

def im_minimum(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = fmin(im1, im2)
    write_fits(outputfile, finalheader, outputimg)
    return finalheader, outputimg

def minmax(data):
    return float(nanmin(data)), float(nanmax(data))

def im_subtract(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
        
    outputimg = im1.data_array - im2.data_array
    write_fits(outputfile, finalheader, outputimg)
    return finalheader, outputimg

def medcombine(fitslist, outputfile = None):
    headers, images = grab_image_stack(fitslist)
    finalheader = headers[0]
    scale_ref = images[0]
    
    for i, im in enumerate(images[1:]):
        #linear fit to find scale factors
        p_init = fm.Linear1D(slope=1, intercept=0)
        fit = fitting.NonLinearLSQFitter()
        p = fit(p_init, im.flatten(), scale_ref.flatten())
        images[i] = p(im).reshape(im.shape)
    
    images = array(images)
    medimage = median(images, axis=0)
    write_fits(outputfile, finalheader, medimage)
    return finalheader, medimage

def scale_spec(ref, spec):
    xref, yref = ref
    xspec, yspec = spec
    yint = interp1d(xspec,yspec)(xref)
    p_init = fm.Linear1D(slope=1, intercept=0)
    fit = fitting.NonLinearLSQFitter()
    p = fit(p_init, yint, yref)
    return [xref, p(yint)]

def combine_spectra(specs, method):
    first_pass = np.median(specs, axis=0)
    if method == 'median':
        return first_pass.tolist()
    smooth = medfilt(first_pass, kernel_size=5)
    weights = []
    for s in specs:
        tmp = power(s - smooth, 2)
        tmp = clip(tmp, min=0.5, max=tmp.max())
        weights.append(reciprocal(tmp))
    tot = weights.sum(axis=0)
    weighted = [s * weights[i] / tot for i, s in enumerate(specs)]
    return weighted.sum(axis=0)
    
    
def pair_dithers(ditherlist):
    a = [i for i, x in enumerate(ditherlist) if x == 'A']
    b = [i for i, x in enumerate(ditherlist) if x == 'B']
    na = float(len(a))
    nb = float(len(b))
    aa = [a[i] for i in (arange(nb)*na/nb).astype('int')] if na > nb else a
    bb = [b[i] for i in (arange(na)*na/nb).astype('int')] if nb > na else b
    return zip(aa, bb)
    
def write_fits(outputfile, header, data):
    if not outputfile:
        return
    outfits = fits.PrimaryHDU(data)
    outfits.header = header
    outfits.writeto(outputfile, output_verify='ignore', clobber=True)
    
def gen_colors(n):
    '''generate a list of dissimilar colors'''
    h = random()
    for x in xrange(n):
        h += golden
        h %= 1
        yield hsv_to_rgb(h, 0.99, 0.99)