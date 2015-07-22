'''
    Some robust statistics routines implementing 
    sigma-clipping to remove outliers.
'''


from astropy.stats import sigma_clip, sigma_clipped_stats
import numpy as np
from scipy.interpolate import interp1d, interp2d
from numpy.ma import masked_invalid
from astropy.modeling import fitting

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
        return self.min(), self.max()

    def fit_to_model(self, model, x=None):
        data = self.clipped()
        f = fitting.LevMarLSQFitter()
        if x is None:
            if self.x is None:
                raise ValueError('Independent variable required.')
            x = self.x
        return f(model, x, data)

    def replace_nans(self, inplace=False):
        if inplace:
            self = masked_invalid(self)
        else:
            return masked_invalid(self)

    def interp(self, target):
        f = interp1d(self.x, self.replace_nans(), kind='cubic')
        return f(target)

class Robust2D(RobustData):
    
    def interp(self, target_x, target_y):
        nx, ny = self.shape
        x, y = np.mgrid[0:nx,0:ny]
        f = interp2d(x, y, self.replace_nans(), kind='cubic')
        return f(target_x, target_y)
    
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