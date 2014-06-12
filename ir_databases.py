import re
from imarith import medcombine
from fitsimage import FitsImage
from numpy import dtype, vstack
from astropy.io import fits
from os import path

from collections import namedtuple, Mapping, OrderedDict

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
    tmp.header['FILES'] = flist
    tmp.update_fits()
    return tmp

InstrumentProfile = namedtuple_with_defaults('InstrumentProfile',['instid', 'tracedir', \
    'dimensions', 'headerkeys', 'description'], ['', 'horizontal', (1024,1024), \
    {'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, ''])

ObsRun = namedtuple_with_defaults('ObsRun', ['runid', 'nights'], ['',{}])

ObsNight = namedtuple_with_defaults('ObsNight', ['date','targets','filestub','rawpath',\
    'outpath','calpath','flaton','flatoff','cals'],['',{},'','','','',[],[],[]])
    
ObsTarget = namedtuple_with_defaults('ObsTarget',['targid', 'instrument_id', 'filestring', \
    'notes', 'images', 'dither', 'spectra'], ['','','','',[],[],[]])

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

#def serialize(data):
#    print 'Serializing: ', data
#    if data is None or isinstance(data, (int, long, float, basestring)):
#        return data
#    if isinstance(data, list):
#        return {"py/list": [serialize(val) for val in data]}
#    if isinstance(data, (InstrumentProfile, ObsRun, ObsNight, ObsTarget)):
#        return {"py/collections.namedtuple": {
#            "type":   type(data).__name__,
#            "fields": list(data._fields),
#            "values": {"py/list":[serialize(getattr(data, f)) for f in data._fields]}}}
#    if isinstance(data, tuple):
#        return {"py/tuple": [serialize(val) for val in data]}
#    if isinstance(data, dict):
#        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.iteritems()]}
#    if isinstance(data, FitsImage):
#        tmp = data.__dict__
#        tmp["data_array"] = None
#        return {"py/FitsImage": serialize(tmp)}
#    
#    raise TypeError("Type %s not data-serializable" % type(data))

#def deserialize(data):
#    print 'Deserializing: ', data
#    if "py/dict" in data:
#        return {key:deserialize(val) for key, val in data["py/dict"].iteritems()}
#    if "py/list" in data:
#        return [deserialize(val) for val in data["py/list"]]
#    if "py/tuple" in data:
#        return (deserialize(val) for val in data["py/tuple"])
#    if "py/collections.namedtuple" in data:
#        dct = data["py/collections.namedtuple"]
#        return namedtuple(dct["type"],dct["fields"])(*deserialize(dct["values"]))
#    if "py/FitsImage" in data:
#        f = FitsImage('')
#        f.__dict__.update(data["py/FitsImage"])
#        return f
#    return data
        
    

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
    