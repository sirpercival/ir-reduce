import re
from imarith import medcombine
from fitsimage import FitsImage
from numpy import dtype, vstack
from astropy.io import fits
from os import path

from collections import namedtuple, Mapping

def namedtuple_with_defaults(typename, field_names, default_values=[]):
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

placeholder = '#'
reg = placeholder+'+'

dither_pattern = ['A', 'B', 'B', 'A']

def parse_filestring(filestring, stub):
    if len(re.findall(reg, stub)) != 1:
        raise ValueError("File format is not valid; must use '#' as placeholder only")
    spot = re.search(reg, stub)
    spotlen = str(spot.end() - spot.start())
    base = re.sub(reg, '%0'+spotlen+'d', stub)
    
    files = []
    tmp = re.split('[.,]', filestring)
    for t in tmp:
        f = re.split('-', t)
        if len(f) == 1:
            files.append(int(f))
        else:
            for i in range(len(f) - 1):
                for j in range(int(f[i]), int(f[i+1])+1):
                    files.append(j)
    images = [FitsImage((base + '.fits') % x) for x in files]
    dithers = [dither_pattern[x % 4] for x in range(len(files))]
    
    return images, dithers
    
def image_stack(flist, stub, output = 'imstack.fits'):
    imlist, junk = parse_filestring(flist, stub)
    imlist = [x.fitsfile for x in imlist]
    comb = medcombine(imlist, outputfile = output)
    tmp = FitsImage(output)
    tmp.flist = flist
    return tmp

InstrumentProfile = namedtuple_with_defaults('InstrumentProfile',['instid', 'tracedir', \
    'dimensions', 'headerkeys', 'description'], ['', 'horizontal', (1024,1024), \
    {'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, ''])

ObsRun = namedtuple_with_defaults('ObsRun', ['runid', 'nights'], ['',{}])

ObsNight = namedtuple_with_defaults('ObsNight', ['date','targets','filestub','rawpath',\
    'outpath','calpath','flaton','flatoff','cals'],['',{},'','','','',[],[],[]])
    
ObsTarget = namedtuple_with_defaults('ObsTarget',['targid', 'instrument_id', 'filestring', \
    'night', 'notes', 'images', 'dither', 'spectra'], ['','','',None,'',[],[],[]])

def add_to(data, element):
    if isinstance(data, ObsRun):
        if isinstance(element, ObsNight):
            data.nights[element.date] = element
        else:
            data.nights[element['date']] = ObsNight(**element)
        return True
    elif isinstance(data, ObsNight):
        if not isinstance(element, ObsTarget):
            element = ObsTarget(**element)
        element.images, element.dither = parse_filestring(element.filestring, \
                path.join(data.rawpath, data.filestub))
        data.targets[element.targid] = target
        return True
    else:
        return False
        
def get_from(data, index):
    if isinstance(data, ObsRun):
        return data.nights.get(index, None)
    elif isinstance(data, ObsNight):
        return data.targets.get(index, None)


class ExtractedSpectrum(object):
    def __init__(self, specfile):
        self.file = specfile
        hdul = fits.open(specfile)
        hdu = hdul[0]
        self.header = hdu.header
        self.spec = hdu.data
        if len(self.spec.shape) == 2:
            self.wav = self.spec[0,:]
            self.spec = self.spec[1,:]
        hdu.close()
        
    def update_fits(self):
        data = vstack((self.wav,self.spec)) if self.wav else self.spec
        fits.update(self.file, data, self.header)    
    