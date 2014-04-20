import re
from imarith import medcombine
from fitsimage import FitsImage
import tables as tb
from numpy import dtype
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
    dithers = [dither_pattern[x mod 4] for x in range(len(files))]
    
    return images
    
def image_stack(flist, stub, output = 'imstack.fits'):
    imlist = parse_filestring(flist, stub)
    comb = medcombine(fitsfiles, outputfiles = output)
    tmp = FitsImage(output)
    tmp.flist = flist
    return tmp

def establish_hdf(output_path = '/', obsrunid = 'observing_run'):
    '''Grab the hdf5 file for this path; if it doesn't exist, create it.'''
    try:
        h5file = tb.open_file(output_path + obsrunid + '.h5', \
            mode = 'r+', title = 'Observing Run '+obsrunid)
        return h5file
    except:
        pass
        
    h5file = tb.open_file(output_path + obsrunid + '.h5', \
        mode = 'w', title = 'Observing Run '+obsrunid)
    group = h5file.create_group("/", "obsnight", "Nights of Observing")
    
    #use compound, nested dtypes to create the heirarchy
    # --> lowest level first
        
    
class ObsTarget(object):
    def __init__(self, **kwargs):
        self.id = kwargs.get('id','')
        self.instrument_id = kwargs.get('iid','')
        self.filestring = files
        self.night = kwargs.get('night',None)
        self.notes = ''
        self.images = []
        self.dither = []
        self.spectra = []

class InstrumentProfile(object):
    def __init__(self, **kwargs):
        self.id = id
        self.tracedir = direction
        self.dimensions = dimensions
        self.headerkeys = header
        self.description = description

class ObsRun(object):
    def __init__(self, **kwargs):
        self.id = id
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
        hdu.close()
        if 'CRVAL' in self.header:
            pass #disentangle wav and spec
        
    def update_fits(self):
        fits.update(self.file, self.data, self.header)

class ObsNight(object):
    def __init__(self, **kwargs):
        self.date = date
        self.targets = {}
        self.filestub = filestub
        self.rawpath = rawpath
        self.outpath = outpath
        self.calpath = calpath
        self.flaton = image_stack(flaton, filestub, output=self.date+'-FlatON.fits') if flaton else None
        self.flatoff = image_stack(flatoff, filestub, output=self.date+'-FlatOFF.fits') if flatoff else None
        self.cals = image_stack(cals, filestub, output=self.date+'-Wavecal.fits') if cals else None
    
    def add_target(self, **kwargs):
        tmp = ObsTarget(**kwargs)
        tmp.files = parse_filestring(self.filestub)
        self.targets[tmp.id] = tmp
    
    