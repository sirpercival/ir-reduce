# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:25:48 2015

@author: gray
"""

import numpy as np
cimport numpy as np
cimport cython

from astropy.modeling import models
from scipy.ndimage.interpolation import geometric_transform

from image_arithmetic import im_subtract, im_minimum
from datatypes import RobustData, Robust2D

cpdef class DitherPair:
    cdef __init__(self):
        pass