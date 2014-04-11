from astropy.io import fits
from astropy.modeling import functional_models as fm, fitting
from numpy import *

def grab_image_stack(imlist):
    images = []
    headers = []
    for ff in imlist:
        hdu = fits.open(fitsfile)
        hdu.close()
        hdu = hdu[0]
        headers.append(hdu.header)
        images.append(hdu.data)
    return headers, images
    
def im_add(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = im1.data + im2.data
    im_write(outputfile, outputimage, finalheader)
    return finalheader, outputimg

def im_divide(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = im1.data / im2.data
    im_write(outputfile, outputimage, finalheader)
    return finalheader, outputimg

def im_subtract(im1, im2, outputfile = None):
    if im1.header:
        finalheader = im1.header
    outputimg = im1.data - im2.data
    im_write(outputfile, outputimage, finalheader)
    return finalheader, outputimg

def medcombine(fitslist, outputfile = None):
    headers, images = grab_image_stack(fitslist)
    finalheader = headers[0]
    scale_ref = images[0]
    
    for i, im in enumerate(images[1:]):
        #linear fit to find scale factors
        p_init = fm.Linear1D(slope=1, intercept=0)
        fit = fitting.NonLinearLSQFitter()
        p = fit(p_init, im, scale_ref)
        images[i] = p(im)
    
    images = array(images)
    medimage = median(images, axis=0)
    im_write(outputfile, medimage, finalheader)
    return finalheader, medimage
    
def pair_dithers(ditherlist):
    a = [i for i, x in enumerate(ditherlist) if x == 'A']
    b = [i for i, x in enumerate(ditherlist) if x == 'B']
    na = float(len(a))
    nb = float(len(b))
    if na > nb:
        aa = [a[i] for i in (arange(nb)*na/nb).astype('int')]
    elif nb > na:
        bb = [b[i] for i in (arange(na)*na/nb).astype('int')]
    return zip(aa, bb)
    
def write_fits(outputfile, header, data):
    if not outputfile:
        return
    outfits = fits.PrimaryHDU(data, header=header)
    outfits.writeto(outputfile)

