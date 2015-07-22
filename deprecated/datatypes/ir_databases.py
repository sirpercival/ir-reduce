from numpy import vstack
from astropy.io import fits
from parse_filestring import parse_filestring
import os

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

InstrumentProfile = namedtuple_with_defaults('InstrumentProfile',['instid', 
    'tracedir', 'dimensions', 'headerkeys', 'description'], ['', 'horizontal', 
    (1024,1024), {'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, ''])

class ObsRun(namedtuple_with_defaults('ObsRun', ['runid', 'nights'], ['',{}])):
    def add_to(self, element):
        if isinstance(element, ObsNight):
            self.nights[element.date] = element
        else:
            self.nights[element['date']] = ObsNight(**element)
        return True
    
    def get_from(self, index):
        return self.nights.get(index, None)

class ObsNight(namedtuple_with_defaults('ObsNight', ['date','targets',
                                                     'filestub','rawpath',
                                                     'outpath','calpath',
                                                     'flaton','flatoff',
                                                     'cals'], ['',{},'','','',
                                                     '',[],[],[]])):
    def add_to(self, element):
        if not isinstance(element, ObsTarget):
            element = ObsTarget(**element)
        element.images, element.dither = parse_filestring(element.filestring, \
                os.path.join(self.rawpath, self.filestub))
        self.targets[element.targid] = element
        return True

    def get_from(self, index):
        return self.targets.get(index, None)
    
ObsTarget = namedtuple_with_defaults('ObsTarget',['targid', 'instrument_id', 'filestring',
    'notes', 'images', 'dither', 'spectra'], ['','','','',[],[],[]])

class ExtractedSpectrum(object):
    def __init__(self, specfile, data=False):
        if data:
            self.spec = specfile
            self.header = None
            self.data_input = True
            self.file = raw_input('Spectrum file name> ')
        else:
            self.file = specfile
            hdul = fits.open(specfile)
            hdu = hdul[0]
            self.header = hdu.header
            self.spec = hdu.data
            self.data_input = False
            hdu.close()
        if len(self.spec.shape) == 2:
            self.wav = self.spec[0,:]
            self.spec = self.spec[1,:]
            
        
    def update_fits(self):
        data = vstack((self.wav,self.spec)) if self.wav else self.spec
        fits.update(self.file, data, self.header)    
    
    def save(self):
        if not self.data_input: self.update_fits()
        data = vstack((self.wav,self.spec)) if self.wav else self.spec
        fits.writeto(self.file, data, header=self.header)
        
        
        
        
        
        
        