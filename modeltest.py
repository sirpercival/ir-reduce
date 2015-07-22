# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:07:38 2015

@author: gray
"""

from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import numpy as np
import json, jsonpickle
import matplotlib.pyplot as plt
from calib import centroids, c_correlate
import pdb


pix0, spec = np.loadtxt('/Users/gray/Desktop/spidr/spec.txt')
wav0, synt = np.loadtxt('/Users/gray/Desktop/spidr/synth.txt')
asn0, asn1 = np.loadtxt('/Users/gray/Desktop/spidr/assign.txt')

spec = medfilt(spec, 3)
fit = fitting.LevMarLSQFitter()

poly = fit(models.Polynomial1D(2), asn0, asn1)
wav1 = poly(pix0)

def trim(arr, bound=wav1):
    return np.logical_and(np.greater_equal(arr, bound.min()), 
                          np.less_equal(arr, bound.max()))

with open('/Users/gray/IRREDUC/storage/linelists.json') as f:
    g = json.load(f)
    g = jsonpickle.decode(g[g.keys()[0]])

lstr = np.array(g['strength'])
lwav = np.array(g['wavelength']) / 1.e4
ok = np.greater_equal(lstr, 100.)
lstr = lstr[ok]
lwav = lwav[ok]
ok = trim(lwav)
lwav = lwav[ok]
lstr = lstr[ok]

inter = interp1d(wav1, pix0)
pix1 = inter(lwav)

iy = interp1d(wav0, synt)(wav1)
adj = spec - medfilt(spec, 51)
lag = np.linspace(-100,100,num=1201,dtype=np.float64)
cc = c_correlate(spec, iy, lag)

pdb.set_trace()

centr = centroids(pix0, spec, tuple(pix1), chisq=True, wid=10.)

plt.clf()
a = np.load('/Users/gray/Desktop/spidr/debug.npy')
for i, (x, y0, y1) in enumerate(a):
    w = x.nonzero()
    plt.plot(x[w], y0[w], 'r-')
    plt.plot(x[w], y1[w], 'b--')
    plt.savefig('/Users/gray/Desktop/spidr/{}.png'.format(i))
    plt.clf()

orig, cen, chi = zip(*centr)

lasn = np.array([lwav[np.where(pix1 == x)] for x in orig]).squeeze()
cen = np.array(cen)

np.savetxt('/Users/gray/Desktop/spidr/asn.txt', np.vstack((cen, lasn)))

poly2 = fit(models.Polynomial1D(2), np.array(cen), np.array(lasn))

wav2 = poly2(pix0)

ok = trim(wav0, bound=wav2)

plt.plot(wav2, spec/spec.max())
plt.plot(wav0[ok], synt[ok]/synt[ok].max(), '--r')
plt.savefig('/Users/gray/Desktop/modeltest.png')