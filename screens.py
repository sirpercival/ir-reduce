# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:40:24 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.properties import (AliasProperty, BooleanProperty, DictProperty, 
                             ListProperty, NumericProperty, ObjectProperty, 
                             StringProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen
from kivy.garden.graph import MeshLinePlot
from kivy.graphics.vertex_instructions import Line, Rectangle
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import numpy as np
from astropy.io import fits
from scipy.constants import golden
from scipy.signal import medfilt
import os, re, glob#, copy
from colorsys import hsv_to_rgb
from random import random
from threading import Thread
import pdb

from image_arithmetic import im_subtract, im_divide
from dialogs import (AddTarget, AlertDialog, AssignLines, DirChooser, 
                     DefineTrace, ExamineSpectrum, FitsHeaderDialog, 
                     SetFitParams, WaitingDialog)
from custom_widgets import default_image, ComboEdit
from datatypes import (interp, twod_to_oned, replace_nans, fitsimage,
                       ExtractedSpectrum, ImageStack, InstrumentProfile, 
                       ObsNight, ObsRun, ObsTarget, scalable_image, 
                       SpectrumStack)
#from datatypes import (ExtractedSpectrum, FitsImage, InstrumentProfile,
#                       ObsNight, ObsRun, ObsTarget, parse_filestring, 
#                       Robust2D, RobustData, SpectrumStack)
from persistence import AdHocDB, instrumentdb, linelistdb, obsrundb, tracedir
from fits_class import (findpeaks1d, fitpeaks1d, #findpeaks2d, fitpeaks2d,
                        extraction, Extraction, CompositeModel)#, extraction_from_state)
from calib import calibrate_wavelength, synth_from_range, detrend

Builder.load_file('screens.kv')

class IRScreen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(IRScreen, self).add_widget(*args)

def gen_colors(n):
    '''generate a list of dissimilar colors'''
    h = random()
    for x in xrange(n):
        h += golden
        h %= 1
        yield hsv_to_rgb(h, 0.99, 0.99)
        
class SpecscrollInsert(BoxLayout):
    active = BooleanProperty(True)
    text = StringProperty('')
    spectrum = ObjectProperty(None)
    index = NumericProperty(0)

class CombineScreen(IRScreen):
    speclist = ListProperty([])
    paths = DictProperty({})
    wmin = NumericProperty(0)
    wmax = NumericProperty(1024)
    dmin = NumericProperty(0)
    dmax = NumericProperty(1024)
    combined_spectrum = ObjectProperty(MeshLinePlot(color=[1,1,1,1]))
    the_specs = ListProperty([])
    theapp = ObjectProperty(None)
    spec_inserts = ListProperty([])
    comb_method = StringProperty('median')
    scaled_spectra = ListProperty([])
    spec_stack = ObjectProperty(None)
    
    def on_enter(self):
        self.speclist = [re.sub('.fits', '', os.path.basename(x)) for x in self.theapp.current_target.images.stack_list]
        flist = [x for y in self.speclist for x in glob.iglob(os.path.join(self.paths['out'],y + '-ap*.fits'))]
        self.the_specs = [ExtractedSpectrum(x) for x in flist]
        self.spec_stack = SpectrumStack(list(self.the_specs))
        colors = list(gen_colors(len(self.the_specs)))
        for i, ts in enumerate(self.the_specs):
            tmp = SpecscrollInsert(text=os.path.basename(flist[i]).replace('.fits',''), 
                                   index=i, spectrum=MeshLinePlot(color=colors[i]+(1,)))
            tmp.bind(active=self.toggle_spectrum)
            if not 'WAVECAL0' in ts.header:
                print i, 
                tmp.active = False
                self.scaled_spectra.append([xrange(len(ts.spec)),ts.spec])
                tmp.spectrum.points = zip(*self.scaled_spectra[i])
                self.spec_inserts.append(tmp)
                continue
            self.scaled_spectra.append([ts.wav,ts.spec])
            tmp.spectrum.points = zip(*self.scaled_spectra[i])
            #tmp.bind(active=lambda *x: self.toggle_spectrum(i))
            self.ids.multispec.add_plot(tmp.spectrum)
            self.spec_inserts.append(tmp)
            self.ids.specscroll.add_widget(tmp)
        self.setminmax()
        self.property('comb_method').dispatch(self)
        if not self.combined_spectrum in self.ids.combspec.plots:
            self.ids.combspec.add_plot(self.combined_spectrum)
        print self.combined_spectrum.color, self.combined_spectrum.points
        
    def setminmax(self):
        mmx = [map(float, [ts.wav.min(), ts.wav.max(), ts.spec.min(), ts.spec.max()]) \
            for i, ts in enumerate(self.the_specs) if self.spec_inserts[i].active]
        mmx = zip(*mmx)
        self.wmin, self.wmax = min(mmx[0]), max(mmx[1])
        self.dmin, self.dmax = min(mmx[2]), max(mmx[3])
    
    def toggle_spectrum(self, instance, active):
        if active:
            self.ids.multispec.add_plot(instance.spectrum)
        else:
            self.ids.multispec.remove_plot(instance.spectrum)
        self.property('comb_method').dispatch(self)
        self.setminmax()
    
    def set_scale(self, spec):
        self.ind = self.speclist.index(spec)
        self.spec_stack = SpectrumStack(list(self.the_specs))
        self.spec_stack.scale(index=self.ind)
        for i, s in enumerate(self.the_specs):
            self.spec_inserts[i].spectrum.points = zip(self.spec_stack[i].wav, self.spec_stack[i].spec)
        self.property('comb_method').dispatch(self)
        self.setminmax()
    
    def on_comb_method(self, instance, value):
        specs = self.spec_stack.subset([x.active for x in self.spec_inserts])
        comb = specs.combine(median=value.lower() == 'median')
        self.combined_spectrum.points = zip(comb.wav, comb.spec)
    
    def combine(self):
        out = self.ids.savefile.text
        h = self.the_specs[self.ind].header
        fits.writeto(out, zip(*self.combined_spectrum.points), header=h)

class ExtractRegionScreen(IRScreen):
    paths = DictProperty({})
    extract_pairs = ListProperty([])
    pairstrings = ListProperty([])
    imwid = NumericProperty(1024)
    imht = NumericProperty(1024)
    bx1 = NumericProperty(0)
    bx2 = NumericProperty(1024)
    by1 = NumericProperty(0)
    by2 = NumericProperty(1024)
    imcanvas = ObjectProperty(None)
    current_extraction = ObjectProperty(None, allownone=True)
    current_flats = ObjectProperty(None, force_dispatch=True, allownone=True)
    current_target = ObjectProperty(None)
    theapp = ObjectProperty(None)
    
    def __init__(self):
        super(ExtractRegionScreen, self).__init__()
        self.ids.ipane.load_data(default_image)
        with self.imcanvas.canvas.after:
            Color(30./255., 227./255., 224./255.)
            self.regionline = Line(points=self.lcoords, close=True, \
                dash_length = 2, dash_offset = 1)

    def on_enter(self):
        flat = os.path.join(self.paths['cal'],'Flat.fits')
        header = [0]
        if not os.path.exists(flat):
            if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
                fon = fitsimage(str(os.path.join(self.paths['cal'], self.theapp.current_night.flaton[0])), header)
                if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
                    foff = fitsimage(str(os.path.join(self.paths['cal'], self.theapp.current_night.flatoff[0])), header)
                    im_subtract(fon, foff, outputfile = flat)
                else:
                    fits.writeto(flat, fon, header=header[0])
        self.current_flats = fitsimage(flat, header)
        self.pairstrings = ['{0} - {1}'.format(*map(os.path.basename,x)) for x in self.extract_pairs]
    
    def on_pre_leave(self):
        self.theapp.current_flats = self.current_flats
        self.theapp.current_extraction = self.current_extraction
    
    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        header = [[0],[0]]
        pair = self.extract_pairs[self.pairstrings.index(val)]
        self.current_extraction = self.current_target.extractions.get(str(hash(':'.join(pair))), None)
        if not self.current_extraction:
            im1, im2 = [fitsimage(os.path.join(self.paths['raw'], x), header[i]) for i, x in enumerate(pair)]
            if 'EXREGX1' in header[0] or 'EXREGX1' in header[1]:
                for x in ['x1','y1','x2','y2']:
                    tmp = header[0]['EXREG'+x.upper()] or header[1]['EXREG'+x.upper()]
                    if tmp:
                        self.set_coord(x, tmp)
            region = map(int,[self.bx1, self.by1, self.bx2, self.by2])
            td = int(tracedir(self.current_target.instrument_id) == 'horizontal')
            self.current_extraction = extraction(im1, im2, self.current_flats, region,
                                                 td, 'Gaussian')
            self.current_extraction.file1 = pair[0]
            self.current_extraction.file2 = pair[1]
            self.current_extraction.flatfile = os.path.join(self.paths['cal'],'Flat.fits')
            self.theapp.current_target.extractions[self.current_extraction.name] = self.current_extraction
        else:
            self.bx1, self.by1, self.bx2, self.by2 = self.current_extraction.region
        im = scalable_image(self.current_extraction.diff)
        self.ids.ipane.load_data(im)
        self.imwid, self.imht = im.dimensions
        
    
    def get_coords(self):
        xscale = float(self.imcanvas.width) / float(self.imwid)
        yscale = float(self.imcanvas.height) / float(self.imht)
        x1 = float(self.bx1) * xscale
        y1 = float(self.by1) * yscale
        x2 = float(self.bx2) * xscale
        y2 = float(self.by2) * yscale
        return [x1, y1, x1, y2, x2, y2, x2, y1] 
    
    lcoords = AliasProperty(get_coords, None, bind=('bx1', 'bx2', 'by1', 'by2', 'imwid', 'imht'))
    
    def set_coord(self, coord, value):
        setattr(self, 'b'+coord, value)
        self.regionline.points = self.lcoords

    def save_region(self):
        self.current_extraction.region = map(int,[self.bx1, self.by1, self.bx2, self.by2])
        self.current_extraction.rebuild()
        self.theapp.current_extraction = self.current_extraction
        self.theapp.save_current()

class InstrumentScreen(IRScreen):
    saved_instrument_names = ListProperty([])
    saved_instruments = ListProperty([])
    instrument_list = ListProperty([])
    current_text = StringProperty('')
    current_instrument = ObjectProperty(InstrumentProfile())
    trace_direction = StringProperty('horizontal')
    
    def on_pre_enter(self):
        self.saved_instrument_names = sorted(instrumentdb.keys())
        self.saved_instruments = [instrumentdb[s] for s in self.saved_instrument_names]
        self.instrument_list = [Button(text = x, size_hint_y = None, height = '30dp') \
            for x in self.saved_instrument_names]
    
    def set_instrument(self):
        self.current_text = self.ids.iprof.text
        try:
            ind = self.saved_instrument_names.index(self.current_text)
            self.current_instrument = self.saved_instruments[ind]
        except ValueError:
            self.current_instrument = InstrumentProfile(instid=self.current_text)
        self.ids.trace_h.state = 'down' if self.current_instrument.tracedir == 'horizontal' else 'normal'
        self.ids.trace_v.state = 'down' if self.current_instrument.tracedir == 'vertical' else 'normal'
        self.ids.xdim.text = str(self.current_instrument.dimensions[0])
        self.ids.ydim.text = str(self.current_instrument.dimensions[1])
        self.ids.idesc.text = self.current_instrument.description
        self.ids.etime.text = self.current_instrument.headerkeys['exp']
        self.ids.secz.text = self.current_instrument.headerkeys['air']
        self.ids.itype.text = self.current_instrument.headerkeys['type']
    
    def save_instrument(self):
        args = {'instid':self.current_text, 'dimensions':(int(self.ids.xdim.text), 
            int(self.ids.ydim.text)), 'tracedir':'horizontal' \
            if self.ids.trace_h.state == 'down' else 'vertical', \
            'description':self.ids.idesc.text, 'headerkeys':{'exp':self.ids.etime.text, \
                'air':self.ids.secz.text, 'type':self.ids.itype.text}}
        new_instrument = InstrumentProfile(**args)
        self.current_instrument = new_instrument
        instrumentdb[new_instrument.instid] = new_instrument
        self.on_pre_enter()

class ObsfileInsert(BoxLayout):
    #obsfile = ObjectProperty(None)
    obsfile = StringProperty('')
    dithertype = StringProperty('')
    header = ObjectProperty(None)

    def launch_header(self):
        head = [0]
        fitsimage(self.obsfile, head)
        self.header_viewer = FitsHeaderDialog(fitsheader = head[0])
        self.header_viewer.bind(on_dismiss = self.update_header())
        self.header_viewer.open()
    
    def update_header(self):
        self.header = self.header_viewer.fitsheader

class ObservingScreen(IRScreen):
    obsids = DictProperty({})
    obsrun_list = ListProperty([])
    obsrun_buttons = ListProperty([])
    current_obsrun = ObjectProperty(ObsRun())
    obsnight_list = ListProperty([])
    obsnight_buttons = ListProperty([])
    current_obsnight = ObjectProperty(ObsNight())
    instrument_list = ListProperty([])
    caltype = StringProperty('')
    target_list = ListProperty([])
    current_target = ObjectProperty(ObsTarget())
    file_list = ListProperty([])
    
    def __init__(self, **kwargs):
        super(ObservingScreen, self).__init__(**kwargs)
        self.rdb = None
        self.waiting = WaitingDialog()
    
    def on_enter(self):
        self.instrument_list = instrumentdb.keys()
        self.obsids = {x:obsrundb[x] for x in obsrundb}
        self.obsrun_list = obsrundb.keys()
        self.obsrun_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsrun_list]
    
    def on_pre_leave(self):
        self.theapp.extract_pairs = [[self.current_target.images.stack_list[x] for x in y] for y in self.current_target.ditherpairs]
        self.theapp.current_target = self.current_target
        self.theapp.current_paths = {'cal':self.current_obsnight.calpath, 
            'raw':self.current_obsnight.rawpath, 'out':self.current_obsnight.outpath}
        self.theapp.current_night = self.current_obsnight
        self.theapp.rdb = self.rdb
    
    def set_obsrun(self):
        if not self.ids.obsrun.text: return
        self.waiting.text = 'Please wait while that observing run loads, thank you!'
        self.waiting.open()
        t = Thread(target=self.obsrun_wrapper)
        t.start()
    
    def obsrun_wrapper(self):
        run_id = self.ids.obsrun.text
        if not run_id: return
        try:
            if run_id not in self.obsids:
                self.rdb = AdHocDB()
                self.obsids[run_id] = self.rdb.fname
                obsrundb[run_id] = self.rdb.fname
            else:
                self.rdb = AdHocDB(self.obsids[run_id])
            self.current_obsrun = ObsRun(runid=run_id)
            self.current_obsrun.nights = {str(self.rdb[r].date):self.rdb[r] for r in self.rdb}
            self.obsnight_list = self.current_obsrun.nights.keys()
            self.obsnight_buttons = [Button(text=x, size_hint_y = None, height = 30) \
                for x in self.obsnight_list]
        except Exception as e:
            print e
            pdb.set_trace()
        self.waiting.dismiss()
    
    def set_obsnight(self):
        night_id = self.ids.obsnight.text
        if night_id == '' or self.current_obsrun.runid == '' \
                        or night_id == 'Observation Date':
            return
        if night_id not in self.obsnight_list:
            self.obsnight_list.append(night_id)
            self.obsnight_buttons.append(Button(text = night_id, \
                size_hint_y = None, height = 30))
            self.current_obsnight = ObsNight(date = night_id)
            self.current_obsrun.add_to(self.current_obsnight)
        else:
            self.current_obsnight = self.current_obsrun.get_from(night_id)
            self.ids.rawpath.text = self.current_obsnight.rawpath
            self.ids.outpath.text = self.current_obsnight.outpath
            self.ids.calpath.text = self.current_obsnight.calpath
            self.ids.fformat.text = self.current_obsnight.filestub
            self.set_filelist()
        for night in self.obsnight_list:
            self.rdb[night] = self.current_obsrun.get_from(night)
        self.target_list = self.current_obsnight.targets.keys()
    
    def pick_rawpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('raw',popup.chosen_directory))
        popup.open()
    
    def setpath(self, which, _dir):
        if which == 'raw':
            self.current_obsnight.rawpath = str(_dir)
            self.ids.rawpath.text = _dir
            if not self.ids.outpath.text:
                self.setpath('out', os.path.join(_dir, 'out'))
            if not self.ids.calpath.text:
                self.setpath('cal', os.path.join(_dir, 'cals'))
        elif which == 'out':
            self.current_obsnight.outpath = str(_dir)
            self.ids.outpath.text = _dir
        elif which == 'cal':
            self.current_obsnight.calpath = str(_dir)
            self.ids.calpath.text = _dir
        
    def pick_outpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('out',popup.chosen_directory))
        popup.open()
        
    def pick_calpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('cal',popup.chosen_directory))
        popup.open()
    
    def check_filestub(self, stub):
        placeholder = '#'
        reg = placeholder+'+'
        if len(re.findall(reg, stub)) != 1:
            popup = AlertDialog(text = "File format is not valid; must use '#' as placeholder only")
            popup.open()
            return
        self.current_obsnight.filestub = stub
    
    def set_caltype(self, caltype):
        if caltype == 'Flats (lamps ON)':
            cout, flist = self.current_obsnight.flaton if self.current_obsnight.flaton else ('Not yet created', '')
        elif caltype == 'Flats (lamps OFF)':
            cout, flist = self.current_obsnight.flatoff if self.current_obsnight.flatoff else ('Not yet created', '')
        elif caltype == 'Arc Lamps':
            cout, flist = self.current_obsnight.cals if self.current_obsnight.cals else ('Not yet created', '')
        self.ids.calfiles.text = flist
        self.ids.calout.text = cout
    
    def set_calfile(self, flist):
        caltype = self.ids.caltypes.text[-2]
        if caltype == 'e': return
        flatfile = os.path.join(self.current_obsnight.calpath,
                                self.current_obsnight.date + {'N':'-FlatON',
                                                              'F':'-FlatOFF',
                                                              'p':'-Wavecal'}[caltype]+'.fits')
        tmp = {'N':self.current_obsnight.flaton, 
               'F':self.current_obsnight.flatoff, 
               'p':self.current_obsnight.cals}[caltype]
        if tmp:
            tmp[1] = flist
        else:
            tmp = ['',flist]
        try:
            header = [0]
            fitsimage(flatfile, header)
            header = header[0]
            if header['FILES'] == flist:
                tmp[0] = flatfile
                self.ids.calout.txt = flatfile
        except:
            pass
        setattr(self.current_obsnight, {'N':'flaton', 'F':'flatoff', 'p':'cals'}[caltype], tmp)
        
    def make_cals(self):
        if not self.current_obsnight.rawpath:
            return
        caltype = self.ids.caltypes.text
        flist = self.ids.calfiles.text
        self.waiting.text = 'Please wait while the calibration images build, thank you!'
        self.waiting.open()
        if caltype == 'Flats (lamps ON)':
            t = Thread(target=self.imstack_wrapper, args=(self.current_obsnight.flaton, flist, \
                self.current_obsnight.date+'-FlatON.fits'))
            t.start()
        elif caltype == 'Flats (lamps OFF)':
            t = Thread(target=self.imstack_wrapper, args=(self.current_obsnight.flatoff, flist, \
                self.current_obsnight.date+'-FlatOFF.fits'))
            t.start()
        elif caltype == 'Arc Lamps':
            t = Thread(target=self.imstack_wrapper, args=(self.current_obsnight.cals, flist, \
                self.current_obsnight.date+'-Wavecal.fits'))
            t.start()
            
    def imstack_wrapper(self, target, flist, outp):
        print 'making flats'
        raw = self.current_obsnight.rawpath
        cal = self.current_obsnight.calpath
        stub = self.current_obsnight.filestub
        imstack = ImageStack(flist, os.path.join(raw, stub))
        imstack.medcombine(outputfile=os.path.join(cal, outp))
        print 'flats made'
        target[:] = [outp, flist]
        self.ids.calfiles.text = flist
        self.ids.calout.text = outp
        self.waiting.dismiss()
    
    def save_night(self):
        tmp = self.current_obsrun.nights
        tmp[self.current_obsnight.date] = self.current_obsnight
        self.current_obsrun.nights = tmp
        for night in self.obsnight_list:
            self.rdb[night] = self.current_obsrun.get_from(night)
        self.rdb.store_sync()
        
    def set_target(self):
        target_id = self.ids.targs.text
        self.current_target = self.current_obsnight.targets[target_id]
        self.set_filelist()
    
    def add_target(self):
        popup = AddTarget(instrumentlist = self.instrument_list)
        popup.open()
        popup.bind(on_dismiss = lambda x: self.update_targets(popup.target_args) \
            if popup.target_args else None)

    def update_targets(self, targs):
        targs['images'] = ImageStack(targs['filestring'], 
                                     os.path.join(self.current_obsnight.rawpath,
                                                        self.current_obsnight.filestub))
        targs['dither'] = targs['images'].dithers
        targs['ditherpairs'] = targs['images'].ditherpairs
        self.current_target = ObsTarget(**targs)
        tmp = self.current_obsnight.targets
        tmp[self.current_target.targid] = self.current_target
        self.current_obsnight.targets = tmp
        self.target_list = self.current_obsnight.targets.keys()
        self.ids.targs.text = self.current_target.targid
        self.set_filelist()
        self.rdb[self.current_obsnight.date] = self.current_obsnight
    
    def set_filelist(self):
        self.ids.obsfiles.clear_widgets()
        self.file_list = []
        for f, dither in zip(self.current_target.images.stack_list, self.current_target.dither):
            tmp = ObsfileInsert(obsfile = f, dithertype = dither)
            self.file_list.append(tmp)
            self.ids.obsfiles.add_widget(tmp)
    
    def save_target(self):
        self.current_target.dither = [x.dithertype for x in self.file_list]
        self.current_target.notes = self.ids.tnotes.text
        #just make sure everything is propagating correctly
        self.current_obsnight.targets[self.current_target.targid] = self.current_target
        self.current_obsrun.nights[self.current_obsnight.date] = self.current_obsnight
        for night in self.obsnight_list:
            self.rdb[night] = self.current_obsrun.get_from(night)
        self.target_list = self.current_obsnight.targets.keys()
    
class TelluricScreen(IRScreen):
    '''This isn't implemented yet'''
    pass

def points_to_array(points):
    x, y = zip(*points)
    return np.array(y)

'''def make_region(im1, im2, region, flat = None, telluric = False):
    if not im1.data_array.any():
        im1.load()
    if not im2.data_array.any():
        im2.load()
    reg1 = im1.data_array[region[0]:region[2]+1, region[1]:region[3]+1]
    reg2 = im2.data_array[region[0]:region[2]+1, region[1]:region[3]+1]
    if flat:
        freg = flat.data_array[region[0]:region[2]+1, region[1]:region[3]+1]
        freg /= np.median(freg)
        reg1 /= freg
        reg2 /= freg
    if telluric:
        return RobustData(im_minimum(reg1, reg2)).normalized
    return RobustData(im_subtract(reg1, reg2)).normalized'''

class ApertureSlider(BoxLayout):
    aperture_line = ObjectProperty(None)
    slider = ObjectProperty(None)
    trash = ObjectProperty(None)
    plot_points = ListProperty([])
    tfscreen = ObjectProperty(None)
    
    def fix_line(self, val):
        x, y = zip(*self.plot_points)
        top_y = interp(np.array(y, dtype=np.float64), np.array(x, dtype=np.float64), np.array([val]))
        self.aperture_line.points = [(val, 0), (val, top_y)]

class TracefitScreen(IRScreen):
    paths = DictProperty([])
    itexture = ObjectProperty(Texture.create(size = (2048, 2048)))
    iregion = ObjectProperty(None)
    current_impair = ObjectProperty(None)
    current_extraction = ObjectProperty(None)
    extractregion = ObjectProperty(None, force_dispatch=True, allownone=True)
    tplot = ObjectProperty(MeshLinePlot(color=[1,1,1,1]))
    current_target = ObjectProperty(None)
    pairstrings = ListProperty([])
    apertures = DictProperty({'pos':[], 'neg':[]})
    drange = ListProperty([0,1024])
    tracepoints = ListProperty([])
    trace_axis = NumericProperty(0)
    fit_params = DictProperty({})
    trace_lines = ListProperty([MeshLinePlot(color=[0,0,1,1]),MeshLinePlot(color=[0,1,1,1])])
    current_flats = ObjectProperty(None, force_dispatch=True, allownone=True)
    theapp = ObjectProperty(None)
    trace_info = DictProperty({})
    rtx = ListProperty([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    
    def __init__(self, **kwargs):
        super(TracefitScreen, self).__init__(**kwargs)
        self.waiting = WaitingDialog(text='Please wait while the extraction is calculated, thank you!')
    
    def on_enter(self):
        self.pairstrings = ['{0} - {1}'.format(*map(os.path.basename,y)) for y in self.theapp.extract_pairs]
        header = [0]
        flat = os.path.join(self.paths['cal'],'Flat.fits')
        if not os.path.exists(flat):
            if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
                fon = fitsimage(str(os.path.join(self.paths['cal'], self.theapp.current_night.flaton[0])), header)
                if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
                    foff = fitsimage(str(os.path.join(self.paths['cal'], self.theapp.current_night.flatoff[0])), header)
                    im_subtract(fon, foff, outputfile = flat)
                else:
                    fits.writeto(flat, fon, header=header[0])
        self.current_flats = fitsimage(flat, header)
        self.theapp.current_flats = self.current_flats
        self.pair_index = -1
    
    def on_pre_leave(self):
        if self.current_target: self.theapp.current_target = self.current_target
        if self.current_extraction: self.theapp.current_extraction = self.current_extraction
        self.theapp.save_current()

    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        if self.pairstrings.index(val) == self.pair_index:
            return
        self.clear_pair()
        self.pair_index = self.pairstrings.index(val)
        pair = self.theapp.extract_pairs[self.pair_index]
        extract_state = self.theapp.current_target.extractions.get(str(hash(':'.join(pair))),None)
        if extract_state:
            self.current_extraction = extract_state #extraction_from_state(extract_state, self.current_flats)
        else:
            popup = AlertDialog(text="You have to select an extraction region for this image pair before you can move on to this step.")
            popup.open()
            return
        self.current_impair = scalable_image(self.current_extraction.diff)
        self.trace_axis = int(tracedir(self.current_target.instrument_id) == 'horizontal')

        idata = ''.join(map(chr,self.current_impair.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.current_impair.dimensions)
        self.extractregion = self.current_extraction.extract_region
        reg = self.current_extraction.region[:]
        reg[2] = reg[2] - reg[0]
        reg[3] = reg[3] - reg[1]
        #if self.trace_axis:
        #    self.iregion = self.itexture.get_region(*reg)
        #else:
        #    self.iregion = self.itexture.get_region(reg[1],reg[0],reg[3],reg[2])
        self.iregion = self.itexture.get_region(*reg)
        self.rtx = [0.0] + [[0.0, 1.0][self.trace_axis]]*2 + [1.0, 1.0] + [[1.0, 0.0][self.trace_axis]]*2 + [0.0]
        dims = [[0,0],list(self.extractregion.shape)]
        dims[0][self.trace_axis] = 0.4 * self.extractregion.shape[self.trace_axis]
        dims[1][self.trace_axis] = 0.6 * self.extractregion.shape[self.trace_axis]
        self.tracepoints = twod_to_oned(self.extractregion[dims[0][0]:dims[1][0]+1,
                                                           dims[0][1]:dims[1][1]+1], axis=self.trace_axis)
        points = replace_nans(np.array(self.tracepoints))
        self.tplot.points = zip(np.arange(points.size), points)
        self.tracepoints = points
        self.drange = [float(points.min()), float(points.max())]
        self.ids.the_graph.add_plot(self.tplot)
        if self.current_extraction.name in self.trace_info:
            info = self.trace_info[self.current_extraction.name]
            for trace in info['ap']['pos']:
                self.add_postrace(val=trace)
            for trace in info['ap']['neg']:
                self.add_negtrace(val=trace)
            self.trace_lines = info['lines']
            self.ids.the_graph.add_plot(self.trace_lines[0])
            self.ids.the_graph.add_plot(self.trace_lines[1])
            self.fit_params = info['par']
        else:
            self.trace_info[self.current_extraction.name] = {'ap':{'pos':[], 'neg':[]}, 'lines':[]}
    
    def clear_pair(self):
        if self.current_extraction:
            self.trace_info[self.current_extraction.name] = {
                'ap':{'pos':[t.slider.value for t in self.apertures['pos']],
                      'neg':[t.slider.value for t in self.apertures['neg']]},
                'lines':self.trace_lines,
                'par': self.fit_params}
        for ap in self.apertures['pos']:
            self.remtrace('pos', ap)
        for ap in self.apertures['neg']:
            self.remtrace('neg', ap)
        self.ids.the_graph.remove_plot(self.trace_lines[0])
        self.ids.the_graph.remove_plot(self.trace_lines[1])
        self.trace_lines = [MeshLinePlot(color=[0,0,1,1]),MeshLinePlot(color=[0,1,1,1])]
        self.fit_params = {}
    
    def add_postrace(self, val=None):
        tp = np.ascontiguousarray(self.tracepoints)
        peaks = findpeaks1d(tp, pn='pos')
        new_peak = float(peaks)
        peakheight = interp(tp, np.arange(tp.size, dtype=np.float64), np.array([new_peak]))
        plot = MeshLinePlot(color=[0,1,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot, tfscreen = self)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = val or new_peak
        newspin.trash.bind(on_press = lambda x: self.remtrace('pos',newspin))
        self.ids.postrace.add_widget(newspin)
        self.apertures['pos'].append(newspin)
        
    def add_negtrace(self, val=None):
        tp = np.ascontiguousarray(self.tracepoints)
        peaks = findpeaks1d(tp, pn='neg')
        new_peak = float(peaks)
        peakheight = interp(tp, np.arange(tp.size, dtype=np.float64), np.array([new_peak]))
        plot = MeshLinePlot(color=[1,0,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot, tfscreen = self)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = val or new_peak
        newspin.trash.bind(on_press = lambda x: self.remtrace('neg',newspin))
        self.ids.negtrace.add_widget(newspin)
        self.apertures['neg'].append(newspin)
    
    def remtrace(self, which, widg):
        self.ids.the_graph.remove_plot(widg.aperture_line)
        self.apertures[which].remove(widg)
        if which == 'pos':
            self.ids.postrace.remove_widget(widg)
        else:
            self.ids.negtrace.remove_widget(widg)
        
    def set_psf(self):
        popup = SetFitParams(fit_args = self.fit_params)
        popup.bind(on_dismiss = lambda x: self.setfp(popup.fit_args))
        popup.open()
    
    def setfp(self, args):
        self.fit_params = args
        
    def fit_trace(self):
        if not self.fit_params or self.fit_params['shape'] not in ('Gaussian','Lorentzian'):
            popup = AlertDialog(text='Make sure you set up your fit parameters!')
            popup.open()
            return
        stripe = np.array(self.tracepoints, dtype=np.float64)
        wid = self.fit_params['wid']
        self.current_extraction.mtype = self.fit_params['shape']
        stripex = np.arange(stripe.size, dtype=float)
        pos = tuple((interp(stripe, stripex.astype(np.float64), np.array([x.slider.value], dtype=np.float64)), 
                     x.slider.value, 
                     wid) for x in self.apertures['pos']) + \
              tuple((interp(stripe, stripex.astype(np.float64), np.array([x.slider.value], dtype=np.float64)), 
                     x.slider.value, 
                     wid) for x in self.apertures['neg'])
        for x in self.trace_lines:
            if x in self.ids.the_graph.plots:
                self.ids.the_graph.remove_plot(x)
        if self.fit_params.get('man',False):
            popup = DefineTrace(npos=len(self.apertures['pos']), \
                nneg=len(self.apertures['neg']), imtexture = self.iregion)
            popup.bind(on_dismiss = self.manual_trace(popup.tracepoints))
            popup.open()
            return
        peaks = fitpeaks1d(stripe, pos, model_type=self.fit_params['shape']).individual()
        self.fit_params['model'] = [p.parameters.tolist() for p in peaks]
        self.fit_params['pmodel'] = [p.parameters.tolist() for p in peaks if p.amplitude > 0]
        self.fit_params['nmodel'] = [p.parameters.tolist() for p in peaks if p.amplitude < 0]
        pmod = CompositeModel(tuple(self.fit_params['pmodel']), self.fit_params['shape'])
        nmod = CompositeModel(tuple(self.fit_params['nmodel']), self.fit_params['shape'])
        self.trace_lines[0].points = zip(stripex, pmod(stripex))
        self.trace_lines[1].points = zip(stripex, nmod(stripex))
        self.ids.the_graph.add_plot(self.trace_lines[0])
        self.ids.the_graph.add_plot(self.trace_lines[1])
    
    def fix_distort(self):
        popup = AlertDialog(text='This is not yet functional')
        popup.open()
        return
        if not (self.fit_params.get('model',False)):
            popup = AlertDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        trace = self.current_extraction.fit_trace(self.fit_params['model'], np.empty(0), False)
        self.current_extraction.fix_distortion(trace)
        self.theapp.current_target.extractions[self.current_extraction.name] = self.current_extraction
        self.set_imagepair(self.pairstrings[self.pair_index])
        self.fit_params['model'] = None
        self.theapp.current_extraction = self.current_extraction
        self.theapp.save_current()
        
    
    def manual_trace(self, traces):
        pass #need to figure out how to apply these
    
    def extract_spectrum(self):
        if not self.fit_params.get('model',False):
            popup = AlertDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        self.waiting.open()
        self.lamp = np.empty([2,2], dtype=np.float64)
        self.lamps = False
        if self.theapp.current_night.cals:
            self.lamp = self.theapp.current_night.cals if not self.current_flats \
                else im_divide(self.theapp.current_night.cals, self.current_flats)
            self.lamp = self.lamp[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
            self.lamps = True
        t = Thread(target=self.extract_wrapper)
        t.start()
    
    def extract_wrapper(self):
        #need a calibration, too
        self.current_extraction.fit_trace(self.fit_params['model'], self.lamp, lamps=self.lamps, extract=True)
        self.waiting.dismiss()
        Clock.schedule_once(self.save_spectra, 0.1)
    
    def save_spectra(self, *args, **kw):
        #write uncalibrated spectra to fits files (will update after calibration)
        lamps = self.lamps
        h1, h2 = [0], [0]
        fitsimage(self.current_extraction.file1, h1)
        h1 = h1[0]
        pstub = os.path.join(self.paths['out'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file1)))
        pstub_sky = os.path.join(self.paths['cal'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file1)))
        ext = ('.fits','-sky.fits','-lamp.fits')
        h1['EXREGX1'] = (self.current_extraction.region[0], 'extraction region coordinate X1')
        h1['EXREGY1'] = (self.current_extraction.region[1], 'extraction region coordinate Y1')
        h1['EXREGX2'] = (self.current_extraction.region[2], 'extraction region coordinate X2')
        h1['EXREGY2'] = (self.current_extraction.region[3], 'extraction region coordinate Y2')
        fitsimage(self.current_extraction.file2, h2)
        h2 = h2[0]
        nstub = os.path.join(self.paths['out'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file2)))
        nstub_sky = os.path.join(self.paths['cal'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file2)))
        h2['EXREGX1'] = (self.current_extraction.region[0], 'extraction region coordinate X1')
        h2['EXREGY1'] = (self.current_extraction.region[1], 'extraction region coordinate Y1')
        h2['EXREGX2'] = (self.current_extraction.region[2], 'extraction region coordinate X2')
        h2['EXREGY2'] = (self.current_extraction.region[3], 'extraction region coordinate Y2')
        
        for j, typ in enumerate(['spec', 'tell']+['lamp']*lamps):
            specs = self.current_extraction.extracts[typ]
            if len(specs.shape) == 1:
                if self.fit_params['model'][0][0] > 0:
                    if typ == 'spec':
                        popup = ExamineSpectrum(target=self.current_target,
                                                spectrum=specs, header=h1,
                                                outfile=(pstub+ext[j])%0)
                        popup.open()
                    else:
                        fits.writeto((pstub_sky+ext[j])%0, specs, header=h1, clobber=True)
                else:
                    if typ == 'spec':
                        popup = ExamineSpectrum(target=self.current_target,
                                                spectrum=specs, header=h2,
                                                outfile=(nstub+ext[j])%0)
                        popup.open()
                    else:
                        fits.writeto((nstub_sky+ext[j])%0, specs, header=h2, clobber=True)
            else:
                specs = specs.T
                for i, ap in enumerate(specs):
                    if self.fit_params['model'][i][0] > 0:
                        if typ == 'spec':
                            popup = ExamineSpectrum(target=self.current_target,
                                                spectrum=ap, header=h1,
                                                outfile=(pstub+ext[j])%i)
                            popup.open()
                        else:
                            fits.writeto((pstub_sky+ext[j])%i, ap, header=h1, clobber=True)
                    else:
                        if typ == 'spec':
                            popup = ExamineSpectrum(target=self.current_target,
                                                spectrum=ap, header=h2,
                                                outfile=(nstub+ext[j])%i)
                            popup.open()
                        else:
                            fits.writeto((nstub_sky+ext[j])%i, -ap, header=h2, clobber=True)
        self.theapp.current_target = self.current_target
        self.theapp.current_extraction = self.current_extraction
        self.theapp.save_current()
        self.extract_done = False

def floor_round(array, dec=0):
    arr = array * np.power(10., dec)
    arr = np.floor(arr)
    return arr * np.power(10., -dec)

def ceil_round(array, dec=0):
    arr = array * np.power(10., dec)
    arr = np.ceil(arr)
    return arr * np.power(10., -dec)

class WavecalScreen(IRScreen):
    paths = DictProperty([])
    speclist = ListProperty([]) #set via app?
    spec_index = NumericProperty(0)
    current_target = ObjectProperty(None)
    current_spectrum = ObjectProperty(None)
    linelist = DictProperty({})
    linelists = ListProperty([])
    linelist_buttons = ListProperty([])
    assignment = ObjectProperty(None)
    synthspec = ListProperty([], force_dispatch=True)
    
    def on_enter(self):
        self.current_target = self.theapp.current_target
        self.speclist = [re.sub('.fits','',os.path.basename(x)) for x in self.current_target.spectra]
        self.linelists = sorted(linelistdb.keys())
        self.linelist_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.linelists]
    
    def set_spectrum(self, spec):
        self.spec_index = self.speclist.index(spec)
        try:
            tmp = ExtractedSpectrum(str(os.path.join(self.paths['out'],self.speclist[self.spec_index] + '.fits')))
        except IOError:
            popup = AlertDialog(text="You haven't extracted that spectrum yet!")
            popup.open()
            return
        if self.current_spectrum:
            self.ids.specdisplay.remove_plot(self.current_spectrum.plot)
        if not self.theapp.current_night.cals:
            self.ids.lampcal_toggle.state = 'normal'
            self.ids.lampcal_toggle.disabled = True
            self.ids.skycal_toggle.state = 'down'
        self.current_spectrum = tmp
        self.current_spectrum.spec = medfilt(self.current_spectrum.spec, 3)
        self.current_spectrum.plot = MeshLinePlot(color=[.9,1,1,1])
        self.current_spectrum.plot.points = zip(self.current_spectrum.wav, self.current_spectrum.spec)
        self.ids.specdisplay.add_plot(self.current_spectrum.plot)
        self.ids.specdisplay.xmin = float(floor_round(self.current_spectrum.wav.min(), dec=1))
        self.ids.specdisplay.xmax = float(ceil_round(self.current_spectrum.wav.max(), dec=1))
        self.ids.specdisplay.ymin = float(floor_round(self.current_spectrum.spec.min(), dec=1))
        self.ids.specdisplay.ymax = float(ceil_round(self.current_spectrum.spec.max(), dec=1))
        self.ids.specdisplay.xlabel = ['Pixel','Wavelength']['WAVECAL0' in self.current_spectrum.header]
    
    def set_linelist(self, val):
        if not val:
            return
        if val in self.linelists:
            self.linelist = linelistdb[val]
        else:
            lines = {'wavelength':[], 'strength':[]}
            with open(val, 'r') as f:
                for l in f:
                    if l.startswith('#'):
                        continue
                    w, s = map(float,l.split())
                    lines['wavelength'] += [w]
                    lines['strength'] += [s]
            self.linelist = lines
            linelistdb[val] = lines
            self.linelists = sorted(linelistdb.keys())
    
    
    def assign(self):
        if not self.linelist:
            popup = AlertDialog(text="Please select a line list first.")
            popup.open()
            return
        fext = '-lamp.fits' if self.theapp.current_night.cals else '-sky.fits'
        fname = os.path.join(self.paths['cal'], 
                             os.path.basename(self.current_spectrum.filename).replace('.fits', fext))
        cal = ExtractedSpectrum(fname)
        cal.spec = detrend(medfilt(cal.spec, 3))
        srange = [float(self.ids.wmin.text), float(self.ids.wmax.text)]
        self.synthspec = synth_from_range(dict(self.linelist), cal.spec.size, srange)
        self.assignment = AssignLines(spectrum=cal, synth=self.synthspec, exp_lo=srange[0], exp_hi=srange[1])
        self.assignment.open()
    
    def wavecal(self):
        if not self.linelist:
            popup = AlertDialog(text="Please select a line list first.")
            popup.open()
            return
        if not self.assignment.assignment:
            popup = AlertDialog(text='Please designate initial line assignments.')
            popup.open()
            return
        calfile = os.path.join(self.paths['cal'],self.speclist[self.spec_index])
        if self.ids.lampcal_toggle.state == 'down':
            calfile += '-lamp.fits'
        else:
            calfile += '-sky.fits'
        try:
            calib = ExtractedSpectrum(str(calfile))
        except IOError:
            popup = AlertDialog(text="You don't have a calibration of this type...")
            popup.open()
            return
        self.calibration = calibrate_wavelength(medfilt(calib.spec), list(self.synthspec), zip(*self.assignment.assignment))
        try:
            for i, w in enumerate(self.calibration.parameters):
                self.current_spectrum.header['WAVECAL%i'%i] = (w, 'Wavelength calibration coefficient')
            self.current_spectrum.wav = self.calibration(range(len(self.current_spectrum.spec)))
            self.ids.specdisplay.xmin = float(floor_round(self.current_spectrum.wav.min(), dec=1))
            self.ids.specdisplay.xmax = float(ceil_round(self.current_spectrum.wav.max(), dec=1))
            self.ids.specdisplay.xlabel = 'Wavelength'
            self.current_spectrum.plot.points = zip(self.current_spectrum.wav, self.current_spectrum.spec)
            self.save_spectrum()
        except Exception as e:
            print e
    
    def save_spectrum(self):
        self.current_spectrum.update_fits()
