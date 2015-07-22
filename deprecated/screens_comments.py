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
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture

import numpy as np
from astropy.io import fits
from scipy.constants import golden
import os, re, glob#, copy
from colorsys import hsv_to_rgb
from random import random
from threading import Thread
import pdb

from image_arithmetic import im_subtract, im_divide
from dialogs import (AddTarget, AlertDialog, DirChooser, DefineTrace, 
                     FitsHeaderDialog, SetFitParams, WaitingDialog)
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
    spectrum = ObjectProperty(MeshLinePlot())

class CombineScreen(IRScreen):
    speclist = ListProperty([])
    paths = DictProperty({})
    wmin = NumericProperty(0)
    wmax = NumericProperty(1024)
    dmin = NumericProperty(0)
    dmax = NumericProperty(1024)
    combined_spectrum = ObjectProperty(MeshLinePlot(color=[1,1,1,1]))
    the_specs = ListProperty([])
    spec_inserts = ListProperty([])
    comb_method = StringProperty('median')
    scaled_spectra = ListProperty([])
    spec_stack = ObjectProperty(None)
    
    def on_enter(self):
        flist = [x for y in self.speclist for x in glob.iglob(self.paths['out'] + y + '-ap*.fits')]
        self.the_specs = [ExtractedSpectrum(x) for x in flist]
        colors = gen_colors(len(self.the_specs))
        for i, ts in enumerate(self.the_specs):
            tmp = SpecscrollInsert(text=flist[i])
            tmp.spectrum.color = colors[i] + (1)
            tmp.bind(active=self.toggle_spectrum(i))
            if not ts.wav:
                tmp.active = False
                self.scaled_spectra.append([xrange(len(ts.spec)),ts.spec])
                tmp.spectrum.points = zip(*self.scaled_spectra[i])
                self.spec_inserts.append(tmp)
                continue
            self.scaled_spectra.append([ts.wav,ts.spec])
            tmp.spectrum.points = zip(*self.scaled_spectra[i])
            tmp.bind(active=self.toggle_spectrum(i))
            self.ids.multispec.add_plot(tmp.spectrum)
            self.spec_inserts.append(tmp)
        self.setminmax()
        self.comb_method.dispatch()
        if not self.combined_spectrum in self.ids.combspec.plots:
            self.ids.combspec.add_plot(self.combined_spectrum)
        
    def setminmax(self):
        mmx = [[ts.wav.min(), ts.wav.max(), ts.spec.min(), ts.spec.max()] \
            for i, ts in enumerate(self.the_specs) if self.spec_inserts[i].active]
        mmx = zip(*mmx)
        self.wmin, self.wmax = min(mmx[0]), max(mmx[1])
        self.dmin, self.dmax = min(mmx[2]), max(mmx[3])
        
    
    def toggle_spectrum(self, ind):
        insert = self.spec_inserts[ind]
        if insert.active:
            self.ids.multispec.add_plot(insert.spectrum)
        else:
            self.ids.multispec.remove_plot(insert.spectrum)
        self.comb_method.dispatch()
        self.setminmax()
    
    def set_scale(self, spec):
        self.ind = self.speclist.index(spec)
        self.spec_stack = SpectrumStack(self.the_specs)
        self.spec_stack.scale(index=self.ind)
        for i, s in enumerate(self.the_specs):
            self.spec_inserts[i].spectrum.points = zip(self.spec_stack[i].x, self.spec_stack[i])
        self.comb_method.dispatch()
        self.setminmax()
    
    def on_comb_method(self, instance, value):
        specs = self.spec_stack.subset([x.active for x in self.spec_inserts.active])
        comb = specs.combine(median=value.lower() == 'median')
        self.combined_spectrum.points = zip(comb.x, comb)
    
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
    current_extraction = ObjectProperty(None)
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
        header = object()
        if not os.path.exists(flat):
            if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
                #fon = FitsImage(os.path.join(self.paths['cal'], \
                #    self.theapp.current_night.flaton[0]), load=True)
                fon = fitsimage(os.path.join(self.paths['cal'], self.theapp.current_night.flaton[0]), header)
                if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
                    #foff = FitsImage(os.path.join(self.paths['cal'], \
                    #    self.theapp.current_night.flatoff[0]), load=True)
                    foff = fitsimage(os.path.join(self.paths['cal'], self.theapp.current_night.flatoff[0]), header)
                    im_subtract(fon, foff, outputfile = flat)
                else:
                    fits.writeto(flat, fon.data_array, header=fon.header)
        self.current_flats = fitsimage(flat, header)
        self.pairstrings = ['{0} - {1}'.format(*map(os.path.basename,x)) for x in self.extract_pairs]
        #self.pairstrings = ['{0} - {1}'.format(*[os.path.basename(x.fitsfile) for x in y]) for y in self.extract_pairs]
    
    def on_pre_leave(self):
        #self.theapp.current_impair = self.current_impair
        self.theapp.current_flats = self.current_flats
        self.theapp.current_extraction = self.current_extraction
    
    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        header = [fits.Header()]*2
        pair = self.extract_pairs[self.pairstrings.index(val)]
        #im1, im2 = [FitsImage(os.path.join(self.paths['raw'], x), 
        #                      load=True) for x in pair]
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
            #pdb.set_trace()
            self.current_extraction = extraction(im1, im2, self.current_flats, region,
                                                 td, 'Gaussian')
            self.current_extraction.file1 = pair[0]
            self.current_extraction.file2 = pair[1]
            self.current_extraction.flatfile = os.path.join(self.paths['cal'],'Flat.fits')
            self.theapp.current_target.extractions[self.current_extraction.name] = self.current_extraction
        else:
            self.bx1, self.by1, self.bx2, self.by2 = self.current_extraction.region
        #fitsfile = self.paths['out']+re.sub(' ','',re.sub('.fits','',val))+'.fits'
        #im1, im2 = [x for x in copy.deepcopy(self.extract_pairs[pair_index])]
        #if not os.path.isfile(fitsfile):
        #    im1, im2 = [x for x in copy.deepcopy(self.extract_pairs[pair_index])]
        #    im1.load(); im2.load()
        #    im_subtract(im1, im2, outputfile=os.path.join(self.paths['out'],fitsfile))
        #self.current_impair = FitsImage(os.path.join(self.paths['out'],fitsfile), load=True)
        #self.ids.ipane.load_data(self.current_impair)
        im = scalable_image(self.current_extraction.diff)
        self.ids.ipane.load_data(im)
        self.imwid, self.imht = im.dimensions
        #self.imwid, self.imht = self.current_impair.dimensions
        #if self.current_impair.get_header_keyword('EXREGX1'):
        #    for x in ['x1', 'y1', 'x2', 'y2']:
        #        tmp = self.current_impair.get_header_keyword('EXREG'+x.upper())
        #        if tmp is not None:
        #            self.set_coord(x, tmp)
        
    
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
        #if coord == 'x1':
        #    self.bx1 = value
        #elif coord == 'x2':
        #    self.bx2 = value
        #elif coord == 'y1':
        #    self.by1 = value
        #elif coord == 'y2':
        #    self.by2 = value
        self.regionline.points = self.lcoords

    def save_region(self):
        #self.current_impair.header['EXREGX1'] = (self.bx1, 'extraction region coordinate X1')
        #self.current_impair.header['EXREGY1'] = (self.by1, 'extraction region coordinate Y1')
        #self.current_impair.header['EXREGX2'] = (self.bx2, 'extraction region coordinate X2')
        #self.current_impair.header['EXREGY2'] = (self.by2, 'extraction region coordinate Y2')
        #self.current_impair.update_fits()
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

    def launch_header(self):
        self.header_viewer = FitsHeaderDialog(fitsimage = self.obsfile)
        self.header_viewer.bind(on_dismiss = self.update_header())
        self.header_viewer.open()
    
    def update_header(self):
        self.obsfile.header = self.header_viewer.fitsimage.header

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
        self.waiting = WaitingDialog(text='Please wait while the calibration images build, thank you!')
    
    def on_enter(self):
        self.instrument_list = instrumentdb.keys()
        self.obsids = {x:obsrundb[x] for x in obsrundb}
        self.obsrun_list = obsrundb.keys()
        self.obsrun_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsrun_list]
    
    def on_pre_leave(self):
        #pairs = self.current_target.ditherpairs
        #self.theapp.extract_pairs = [[self.current_target.images.images[x] for x in y] for y in pairs]
        #self.theapp.extract_pairs = [[os.path.basename(self.current_target.images.stack_list[x]) for x in y] for y in self.current_target.ditherpairs]
        self.theapp.extract_pairs = [[self.current_target.images.stack_list[x] for x in y] for y in self.current_target.ditherpairs]
        self.theapp.current_target = self.current_target
        self.theapp.current_paths = {'cal':self.current_obsnight.calpath, 
            'raw':self.current_obsnight.rawpath, 'out':self.current_obsnight.outpath}
        self.theapp.current_night = self.current_obsnight
        self.theapp.rdb = self.rdb
    
    def set_obsrun(self):
        run_id = self.ids.obsrun.text
        if not run_id: return
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
            self.current_obsnight.rawpath = _dir
            self.ids.rawpath.text = _dir
        elif which == 'out':
            self.current_obsnight.outpath = _dir
            self.ids.outpath.text = _dir
        elif which == 'cal':
            self.current_obsnight.calpath = _dir
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
            header = object()
            fitsimage(flatfile, header)
            #if FitsImage(flatfile).header['FILES'] == flist:
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
        self.waiting.open()
        if caltype == 'Flats (lamps ON)':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.flaton, flist, \
                self.current_obsnight.date+'-FlatON.fits'))
            t.start()
        elif caltype == 'Flats (lamps OFF)':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.flatoff, flist, \
                self.current_obsnight.date+'-FlatOFF.fits'))
            t.start()
        elif caltype == 'Arc Lamps':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.cals, flist, \
                self.current_obsnight.date+'-Wavecal.fits'))
            t.start()
            
    def imstack_wrapper(self, target, flist, outp):
        raw = self.current_obsnight.rawpath
        cal = self.current_obsnight.calpath
        stub = self.current_obsnight.filestub
        #imstack = parse_filestring(flist, os.path.join(raw, stub), preserve=True)
        imstack = ImageStack(flist, os.path.join(raw, stub))
        imstack.medcombine(outputfile = os.path.join(cal, outp))
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
        #targs['images'] = parse_filestring(targs['filestring'],
        targs['images'] = ImageStack(targs['filestring'], 
                                     os.path.join(self.current_obsnight.rawpath,
                                                        self.current_obsnight.filestub))
        #                                   os.path.join(self.current_obsnight.rawpath,
        #                                                self.current_obsnight.filestub), 
        #                                   preserve=True)
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
        #y = RobustData(y, x=x)
        #top_y = y.interp(val)
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
    
    def on_enter(self):
        #self.pairstrings = ['{0} - {1}'.format(*[os.path.basename(x.fitsfile) for x in y]) \
        #    for y in self.theapp.extract_pairs]
        self.pairstrings = ['{0} - {1}'.format(*map(os.path.basename,y)) for y in self.theapp.extract_pairs]
        header = object()
        flat = os.path.join(self.paths['cal'],'Flat.fits')
        if not os.path.exists(flat):
            if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
                #fon = FitsImage(os.path.join(self.paths['cal'], \
                #    self.theapp.current_night.flaton[0]), load=True)
                fon = fitsimage(os.path.join(self.paths['cal'], self.theapp.current_night.flaton[0]), header)
                if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
                    #foff = FitsImage(os.path.join(self.paths['cal'], \
                    #    self.theapp.current_night.flatoff[0]), load=True)
                    foff = fitsimage(os.path.join(self.paths['cal'], self.theapp.current_night.flatoff[0]), header)
                    im_subtract(fon, foff, outputfile = flat)
                else:
                    fits.writeto(flat, fon.data_array, header=fon.header)
        self.current_flats = fitsimage(flat, header)
        self.theapp.current_flats = self.current_flats
        #if not self.theapp.current_flats and self.theapp.current_night.flaton:
        #    flat = os.path.join(self.paths['cal'],'Flat.fits')
        #    try:
        #        self.current_flats = FitsImage(flat, load = True)
        #        self.theapp.current_flats = self.current_flats
        #    except:
        #        if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
        #            fon = FitsImage(os.path.join(self.paths['cal'], \
        #                self.theapp.current_night.flaton[0]), load=True)
        #            if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
        #                foff = FitsImage(os.path.join(self.paths['cal'], \
        #                    self.theapp.current_night.flatoff[0]), load=True)
        #                im_subtract(fon, foff, outputfile = flat)
        #            else:
        #                fits.writeto(flat, fon.data_array, header=fon.header)
        #            self.current_flats = FitsImage(flat, load = True)
        #        self.theapp.current_flats = self.current_flats
        #else:
        #    self.current_flats = self.theapp.current_flats

    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        self.pair_index = self.pairstrings.index(val)
        pair = self.theapp.extract_pairs[self.pair_index]
        extract_state = self.theapp.current_target.extractions.get(str(hash(':'.join(pair))),None)
        if extract_state:
            self.current_extraction = extract_state #extraction_from_state(extract_state, self.current_flats)
        else:
            popup = AlertDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        
        #fitsfile = self.paths['out']+re.sub(' ','',re.sub('.fits','',val))+'.fits'
        #if not os.path.isfile(fitsfile):
        #    popup = AlertDialog(text='You have to select an extraction'\
        #        'region for this image pair \nbefore you can move on to this step.')
        #    popup.open()
        #    return
        #self.current_impair = FitsImage(fitsfile)
        #self.region = self.current_impair.get_header_keyword(*('EXREG' + x for x in ['X1','Y1','X2','Y2']))
        #if not any(self.region):
        #    popup = AlertDialog(text='You have to select an extraction'\
        #        'region for this image pair \nbefore you can move on to this step.')
        #    popup.open()
        #    return
        self.current_impair = scalable_image(self.current_extraction.diff)
        idata = ''.join(map(chr,self.current_impair.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.current_impair.dimensions)
        #self.trace_axis = 0 if tracedir(self.current_target.instrument_id) == 'vertical' else 1
        self.trace_axis = int(tracedir(self.current_target.instrument_id) == 'horizontal')
        #if self.trace_axis:
            #self.trace_axis = 1
        #    reg = [self.region[x] for x in [1, 0, 3, 2]]
            #self.extractregion = make_region(pair[0], pair[1], reg, self.current_flats)#.transpose()
        #else:
        #    self.extractregion = make_region(pair[0], pair[1], self.region, self.current_flats).transpose()
        #self.current_extraction = extraction(pair[0].data, pair[0].data, self.current_flats.data,
        #                                     self.region, self.trace_axis, 'Gaussian')
        self.extractregion = self.current_extraction.extract_region
        reg = self.current_extraction.region[:]
        reg[2] = reg[2] - reg[0]
        reg[3] = reg[3] - reg[1]
        self.iregion = self.itexture.get_region(*reg)
        dims = [[0,0],list(self.extractregion.shape)]
        dims[0][self.trace_axis] = 0.4 * self.extractregion.shape[self.trace_axis]
        dims[1][self.trace_axis] = 0.6 * self.extractregion.shape[self.trace_axis]
        #self.tracepoints = Robust2D(self.extractregion[dims[0][0]:dims[1][0]+1,
        #                                               dims[0][1]:dims[1][1]+1]).combine(axis=self.trace_axis)
        self.tracepoints = twod_to_oned(self.extractregion[dims[0][0]:dims[1][0]+1,
                                                           dims[0][1]:dims[1][1]+1], axis=self.trace_axis)
        #points = RobustData(self.tracepoints, index=True)
        #points.replace_nans(inplace=True)
        points = replace_nans(np.array(self.tracepoints))
        self.tplot.points = zip(np.arange(points.size), points)
        self.tracepoints = points
        self.drange = [float(points.min()), float(points.max())]
        self.ids.the_graph.add_plot(self.tplot)
    
    def add_postrace(self):
        tp = np.ascontiguousarray(self.tracepoints)
        peaks = findpeaks1d(tp, pn='neg')
        new_peak = float(peaks)
        peakheight = interp(tp, np.arange(tp.size, dtype=np.float64), np.array([new_peak]))#points.interp(new_peak)
        plot = MeshLinePlot(color=[0,1,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot, tfscreen = self)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = new_peak
        newspin.trash.bind(on_press = lambda x: self.remtrace('pos',newspin))
        self.ids.postrace.add_widget(newspin)
        self.apertures['pos'].append(newspin)
        
    def add_negtrace(self):
        tp = np.ascontiguousarray(self.tracepoints)
        peaks = findpeaks1d(tp, pn='neg')
        new_peak = float(peaks)
        peakheight = interp(tp, np.arange(tp.size, dtype=np.float64), np.array([new_peak]))#points.interp(new_peak)
        plot = MeshLinePlot(color=[1,0,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot, tfscreen = self)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = new_peak
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
        #stripe = RobustData(self.tracepoints, index=True)
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
        #im1, im2 = [x for x in copy.deepcopy(self.theapp.extract_pairs[self.pair_index])]
        #im1.load(); im2.load()
        #dist_reshape = lambda x, y: x.reshape(len(y), x.size/len(y))
        #pdistort = dist_reshape(self.fit_params['pmodel'].parameters, self.apertures['pos'])
        #ndistort = dist_reshape(self.fit_params['nmodel'].parameters, self.apertures['neg'])
        #im1.data_array = self.current_extraction.fix_distortion(self.fit_params['model'])
        #im2.data_array = extract.fix_distortion(im2.data_array, ndistort)
        self.current_extraction.fix_distortion(trace)
        self.theapp.current_target.extractions[self.current_extraction.name] = self.current_extraction
        #im_subtract(im1, im2, outputfile=self.current_impair.fitsfile)
        #tmp = self.current_impair
        #self.current_impair = FitsImage(self.current_impair.fitsfile)
        #self.current_impair.header['EXREGX1'] = (tmp.get_header_keyword('EXREGX1'), 'extraction region coordinate X1')
        #self.current_impair.header['EXREGY1'] = (tmp.get_header_keyword('EXREGY1'), 'extraction region coordinate Y1')
        #self.current_impair.header['EXREGX2'] = (tmp.get_header_keyword('EXREGX2'), 'extraction region coordinate X2')
        #self.current_impair.header['EXREGY2'] = (tmp.get_header_keyword('EXREGY2'), 'extraction region coordinate Y2')
        #self.current_impair.update_fits(header_only = True)
        self.set_imagepair(self.pairstrings[self.pair_index])
        self.fit_params['model'] = None
        self.theapp.current_extraction = self.current_extraction
        self.theapp.save_current()
        #self.fit_params['nmodel'] = None
        #self.fit_params['pmodel'] = None
        pdb.set_trace()
        
    
    def manual_trace(self, traces):
        pass #need to figure out how to apply these
    
    def extract_spectrum(self):
        if not self.fit_params.get('model',False):
            popup = AlertDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        
        #need a calibration, too
        self.lamp = np.empty([2,2], dtype=np.float64)
        lamps = False
        if self.theapp.current_night.cals:
            self.lamp = self.theapp.current_night.cals if not self.current_flats \
                else im_divide(self.theapp.current_night.cals, self.current_flats)
            self.lamp = self.lamp[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
            lamps = True
        
        #im1, im2 = [x for x in copy.deepcopy(self.theapp.extract_pairs[self.pair_index])]
        #im1.load(); im2.load()
        #im1.data_array = extract.fix_distortion(im1.data_array, pdistort)
        #im2.data_array = extract.fix_distortion(im2.data_array, ndistort)
        #self.tell = make_region(im1, im2, self.region, flat=self.current_flats, telluric=True)
        #self.pextract = extract(self.fit_params['pmodel'], self.extractregion, self.tell, 'pos', lamp = self.lamp)
        #self.nextract = extract(self.fit_params['nmodel'], self.extractregion, self.tell, 'neg', lamp = self.lamp)
        self.current_extraction.fit_trace(self.fit_params['model'], self.lamp, lamps=lamps, extract=True)
        #if self.lamp.size > 0:
        #    spectra, tellurics, lamps = self.current_extraction.extract(trace, 
        #                                                                self.lamp, 
        #                                                                False)
        #else:
        #    spectra, telluric = self.current_extraction.extract(trace, self.lamp, True)
        
        #write uncalibrated spectra to fits files (will update after calibration)
        h1, h2 = fits.Header(), fits.Header()
        fitsimage(self.current_extraction.file1, h1)
        #im1 = FitsImage(self.current_extraction.file1, load=False)
        pstub = os.path.join(self.paths['out'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file1)))#im1.fitsfile)
        ext = ('.fits','-sky.fits','-lamp.fits')
        #h = im1.header
        h1['EXREGX1'] = (self.current_extraction.region[0], 'extraction region coordinate X1')
        h1['EXREGY1'] = (self.current_extraction.region[1], 'extraction region coordinate Y1')
        h1['EXREGX2'] = (self.current_extraction.region[2], 'extraction region coordinate X2')
        h1['EXREGY2'] = (self.current_extraction.region[3], 'extraction region coordinate Y2')
        fitsimage(self.current_extraction.file2, h2)
        nstub = os.path.join(self.paths['out'], 
                             re.sub('.fits','-ap%i',
                                    os.path.basename(self.current_extraction.file2)))#im2.fitsfile)
        #h = im2.header
        h2['EXREGX1'] = (self.current_extraction.region[0], 'extraction region coordinate X1')
        h2['EXREGY1'] = (self.current_extraction.region[1], 'extraction region coordinate Y1')
        h2['EXREGX2'] = (self.current_extraction.region[2], 'extraction region coordinate X2')
        h2['EXREGY2'] = (self.current_extraction.region[3], 'extraction region coordinate Y2')
        
        for j, typ in enumerate(['spec', 'tell']+['lamp']*lamps):
            specs = self.current_extraction.extracts[typ]
            if len(specs.shape) == 1:
                if self.fit_params['model'][0][0] > 0:
                    fits.writeto((pstub+ext[j])%0, specs, header=h1)
                else:
                    fits.writeto((nstub+ext[j])%0, specs*-1., header=h2)
            else:
                specs = specs.T
                for i, ap in enumerate(specs):
                    if self.fit_params['model'][i][0] > 0:
                        fits.writeto((pstub+ext[j])%i, ap, header=h1)
                    else:
                        fits.writeto((nstub+ext[j])%i, ap*-1., header=h2)
        return
        #
        #for i, p_ap in enumerate(self.pextract):
        #    for j in range(p_ap.shape[1]):
        #        spec = p_ap[:,j]
        #        fits.writeto((pstub + ext[i]) % j, spec, header=h)
        ##im2 = FitsImage(self.current_extraction.file2, load=False)
        #for i, n_ap in enumerate(self.nextract):
        #    for j in range(n_ap.shape[1]):
        #        spec = n_ap[:,j]
        #        fits.writeto((nstub + ext[i]) % j, spec, header=h)

class WavecalScreen(IRScreen):
    paths = DictProperty([])
    wmin = NumericProperty(0)
    wmax = NumericProperty(1024)
    dmin = NumericProperty(0)
    dmax = NumericProperty(1024)
    speclist = ListProperty([]) #set via app?
    spec_index = NumericProperty(0)
    current_spectrum = ObjectProperty(None)
    linelist = StringProperty('')
    linelists = ListProperty([])
    linelist_buttons = ListProperty([])
    
    def on_enter(self):
        self.linelists = linelistdb.keys()
        self.linelist_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in linelistdb]
    
    def set_spectrum(self, spec):
        self.spec_index = self.speclist.index(spec)
        try:
            tmp = ExtractedSpectrum(self.paths['out']+self.speclist[self.spec_index] + '.fits')
        except:
            popup = AlertDialog(text="You haven't extracted that spectrum yet!")
            popup.open()
            return
        if self.current_spectrum:
            self.ids.specdisplay.remove_plot(self.current_spectrum.plot)
        self.current_spectrum = tmp
        self.current_spectrum.plot = MeshLinePlot(color=[.9,1,1,1])
        if not self.current_spectrum.wav:
            self.wmin = 0
            self.wmax = len(self.current_spectrum.spec)-1
            self.current_spectrum.wav = range(self.wmax)
        else:
            self.wmin = self.current_spectrum.wav.min()
            self.wmax = self.current_spectrum.wav.max()
        self.dmin, self.dmax = minmax(self.current_spectrum.spec)
        self.current_spectrum.plot.points = zip(self.current_spectrum.wav, self.current_spectrum.spec)
        self.ids.specdisplay.add_plot(self.current_spectrum.plot)
            
    def set_wmin(self, val):
        self.wmin = val
    
    def set_wmax(self, val):
        self.wmax = val
    
    def set_linelist(self, val):
        self.linelist = val
    
    def wavecal(self):
        if not self.linelist:
            popup = AlertDialog(text="Please select a line list first.")
            popup.open()
            return
        calfile = self.paths['cal'] + self.speclist[self.spec_index]
        if self.ids.lampcal.state == 'down':
            calfile += '-lamp.fits'
        else:
            calfile += '-sky.fits'
        try:
            calib = ExtractedSpectrum(calfile)
        except:
            popup = AlertDialog(text="You don't have a calibration of this type...")
            popup.open()
            return
        niter = self.ids.numiter.text
        if self.linelist in self.linelists:
            #lldb = shelve.open('storage/linelists')
            #linelist_path = ird.deserialize(linelistdb[self.lineslist])
            linelist_path = linelistdb[self.linelist]
            #lldb.close()
        else:
            linelist_path = self.linelist
        self.calibration = calibrate_wavelength(calib, linelist_path, (self.wmin, self.wmax), niter)
        for i, w in self.calibration.parameters:
            self.current_spectrum.header['WAVECAL%i'%i] = (w, 'Wavelength calibration coefficient')
        self.current_spectrum.wav = self.calibration(range(len(self.current_spectrum.spec)))
        self.wmin = self.current_spectrum.wav.min()
        self.wmax = self.current_spectrum.wav.max()
        self.current_spectrum.plot.points = zip(self.current_spectrum.wav, self.current_spectrum.spec)
    
    def save_spectrum(self):
        self.current_spectrum.update_fits()
