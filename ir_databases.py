import re
from imarith import medcombine
from fitsimage import FitsImage
from numpy import dtype, vstack
from astropy.io import fits

placeholder = '#'
reg = placeholder+'+'

dither_pattern = ['A', 'B', 'B', 'A']

def parse_filestring(filestring, stub, dithers = []):
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
    images = [FitsImage(base % x) for x in files]
    dithers = [dither_pattern[x % 4] for x in range(len(files))]
    
    return images
    
def image_stack(flist, stub, output = 'imstack.fits'):
    imlist = parse_filestring(flist, stub)
    comb = medcombine(imlist, outputfile = output)
    tmp = FitsImage(output)
    tmp.flist = flist
    return tmp
    
class ObsTarget(object):
    def __init__(self, **kwargs):
        self.targid = kwargs.get('id','')
        self.instrument_id = kwargs.get('iid','')
        self.filestring = kwargs.get('files','')
        self.night = kwargs.get('night',None)
        self.notes = ''
        self.images = []
        self.dither = []
        self.spectra = []

class InstrumentProfile(object):
    def __init__(self, **kwargs):
        self.instid = kwargs.get('instid','')
        self.tracedir = kwargs.get('direction','horizontal')
        self.dimensions = kwargs.get('dimensions',(1024,1024))
        self.headerkeys = kwargs.get('header', {})
        self.description = kwargs.get('description','')

class ObsRun(object):
    def __init__(self, **kwargs):
        self.runid = kwargs.get('runid','')
        self.nights = {}
    
    def addnight(self, night):
        if type(night) is ObsNight:
            self.nights[night.date] = night
        else:
            self.nights[night['date']] = ObsNight(**night)
    
    def get_night(self, nightid):
        return self.nights.get(nightid, None)

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

class ObsNight(object):
    def __init__(self, **kwargs):
        self.date = kwargs.get('date','')
        self.targets = {}
        self.filestub = kwargs.get('filestub','')
        self.rawpath = kwargs.get('rawpath','')
        self.outpath = kwargs.get('outpath','')
        self.calpath = kwargs.get('calpath','')
        self.flaton = image_stack(flaton, filestub, output=self.date+'-FlatON.fits') if kwargs.get('flaton',False) else None
        self.flatoff = image_stack(flatoff, filestub, output=self.date+'-FlatOFF.fits') if kwargs.get('flatoff',False) else None
        self.cals = image_stack(cals, filestub, output=self.date+'-Wavecal.fits') if kwargs.get('cals',False) else None
    
    def add_target(self, **kwargs):
        tmp = ObsTarget(**kwargs)
        tmp.files = parse_filestring(self.filestub)
        self.targets[tmp.targid] = tmp
    
    