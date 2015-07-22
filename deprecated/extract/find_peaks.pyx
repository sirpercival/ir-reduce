# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:40:01 2015

@author: gray
"""

import numpy as np
cimport numpy as np
cimport cython

from datatypes import RobustData, Robust2D

@cython.boundscheck(False)
cpdef int find_peak_2d(np.ndarray[double, ndim=2] data, int tracedir=-1, 
                       str pn='pos'):
    data = Robust2D(data)
    data.replace_nans()
    tracedir = abs(tracedir) # null tracedir (-1) becomes 2nd axis

    #compress along the trace using a robust mean
    return find_peak_1d(data.combine(axis=tracedir), pn=pn)

@cython.boundscheck(False)
cpdef int find_peak_1d(np.ndarray[double, ndim=1] data, str pn='pos'):
    if not isinstance(data, RobustData):
        data = RobustData(data)
    data.clipped(inplace=True)
    if pn == 'pos':
        return data.argmax()
    else:
        return data.argmin()