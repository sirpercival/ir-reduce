import re
import imarith
from fitsimage import FitsImage

placeholder = '#'
reg = placeholder+'+'

dither_pattern = ['A', 'B', 'B', 'A']


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
    pass

class ObsTarget(object):
    def __init__(self, **kwargs):
        self.id = kwargs.get('id','')
        self.instrument_id = kwargs.get('iid','')
        self.filestring = files
        self.notes = ''
        self.images = []
        self.dither = []
        self.spectra = []
    
    def parse_filestring(self, stub):
        if len(re.findall(reg, stub)) != 1:
            raise ValueError("File format is not valid; must use '#' as placeholder only")
        spot = re.search(reg, stub)
        spotlen = str(spot.end() - spot.start())
        base = re.sub(reg, '%0'+spotlen+'d', stub)
        
        files = []
        tmp = re.split('[.,]', self.filestring)
        for t in tmp:
            f = re.split('-', t)
            if len(f) == 1:
                files.append(int(f))
            else:
                for i in range(len(f) - 1):
                    for j in range(int(f[i]), int(f[i+1])+1):
                        files.append(j)
        self.images = [FitsImage(base % x) for x in files]
        self.dither = [dither_pattern[x mod 4] for x in range(len(files))]
    
    def add_spectrum(self, spec):
        if not isinstance(spec, ExtractedSpectrum): return
        self.spectra.append(spec)

class ObsNight(object):
    def __init__(self, **kwargs):
        self.date = date
        self.targets = {}
        self.filestub = filestub
        self.rawpath = rawpath
        self.outpath = outpath
        self.calpath = calpath
        self.flaton = self.add_stack('flats-on', flaton, output=self.date+'-FlatON.fits') if flaton else None
        self.flatoff = self.add_stack('flats-off', flatoff, output=self.date+'-FlatOFF.fits') if flatoff else None
        self.cals = self.add_stack('cals', cals, output=self.date+'-Wavecal.fits') if cals else None
    
    def add_target(self, **kwargs):
        tmp = ObsTarget(**kwargs)
        tmp.parse_filestring(self.filestub)
        self.targets[tmp.id] = tmp
    
    def add_stack(self, id, flist, output = 'imstack.fits'):
        tmp = ObsTarget(id=id, files=flist)
        tmp.parse_filestring(self.filestub)
        fitsfiles = [x.fitsfile for x in tmp.images]
        comb = medcombine(fitsfiles, outputfiles = output)
        tmp = FitsImage(output)
        tmp.flist = flist
        return tmp
    
    