from astropy.io import fits
import numpy as np 
from scipy.misc import bytescale
from robuststats import array_process
from astropy.modeling import functional_models as fm, fitting

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
    data = np.sort(np.array(data))
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    center_pixel = (num_points + 1) / 2
    if data_min == data_max:
        return data_min, data_max
    med = np.median(data)
    clipped_data = array_process(data, 3., compress=True)
    if clipped_data.size < int(num_points/2.0):
        return data_min, data_max
    x_data = np.arange(clipped_data.size)
    p_init = fm.Linear1D(1, 0)
    fit = fitting.NonLinearLSQFitter()
    p = fit(p_init, x_data, clipped_data)
    z1 = med - (center_pixel-1) * p.slope / contrast
    z2 = med + (num_points - center_pixel) * p.slope / contrast
    zmin = max(z1, data_min)
    zmax = min(z2, data_max)
    if zmin >= zmax:
        return data_min, data_max
    return zmin, zmax

def grab_header(file):
    hdul = fits.open(file)
    header = hdul[0].header
    hdul.close()
    return header

class ScalableImage(object):
    def __init__(self):
        self.data_array = np.array([])
        self.dimensions = [0, 0]
        self.threshold = [0, 0]
        self.factor = 0.
        self.mode = ''

    def load(self, data, scalemode = 'linear', factor = None):
        self.data_array = data
        self.dimensions[1], self.dimensions[0] = self.data_array.shape
        #mn, mx = np.nanmin(self.data_array), np.nanmax(self.data_array)
        #drange = mx - mn
        #self.threshold = [mn + drange * 0.05, mn + drange * 0.95]
        self.threshold = zscale(self.data_array)
        self.factor = factor
        self.mode = scalemode
        self.imstretch()
    
    def change_parameters(self, info):
        self.threshold = [info['min'], info['max']]
        self.mode = info['mode']
        self.factor = info['factor']
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
            tmp = bytescale(data, high=1.)
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
        self.scaled = bytescale(data).flatten().tolist()

class FitsImage(ScalableImage):
    def __init__(self, fitsfile, header = None, load = False):
        super(FitsImage, self).__init__()
        self.fitsfile = fitsfile
        self.header = grab_header(fitsfile) if not header else header
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
    
    def get_header_keyword(self, *args):
        return [self.header.get(x, None) for x in args]


if __name__ == '__main__':
    pass
    