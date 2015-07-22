# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:30:00 2015

@author: gray
"""

from datatypes import Robust2D
from astropy.modeling.models import Polynomial1D
from scipy.ndimage.interpolation import geometric_transform

def fix_distortion(image, centers, tracedir=-1):
    '''Fit the path of the trace with a polynomial, and warp the image
    back to straight.'''
    tracedir = abs(tracedir)
    
    centers = Robust2D(centers)
    centers = (centers.T - centers.T[0,:]).T.combine()
    distortion = centers.fit_to_model(Polynomial1D(degree=2), 
                                      x=centers.index)
    def undistort(coords):
        xp, yp = coords
        if tracedir:
            xp, yp = yp, xp
        if tracedir:
            return yp - distortion(xp), xp
        return xp, yp-distortion(xp)

    return geometric_transform(image, undistort)