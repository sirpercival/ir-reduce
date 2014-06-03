import kivy
kivy.require('1.8.0')

from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.garden.graph import Graph, MeshLinePlot #SmoothLinePlot when upgraded to 1.8.1
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.properties import ListProperty, NumericProperty, ObjectProperty, \
    StringProperty, DictProperty, AliasProperty, BooleanProperty
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from imagepane import ImagePane, default_image
from fitsimage import FitsImage
from comboedit import ComboEdit
import ir_databases as ird
from dialogs import FitsHeaderDialog, DirChooser, AddTarget, SetFitParams, WarningDialog, DefineTrace, WaitingDialog
from imarith import pair_dithers, im_subtract, im_minimum, minmax, write_fits, gen_colors, scale_spec, combine_spectra
from robuststats import robust_mean as robm, interp_x, idlhash, med_normal
from findtrace import find_peaks, fit_multipeak, draw_trace, undistort_imagearray, extract
from calib import calibrate_wavelength
from kivy.storage.jsonstore import JsonStore
import shelve, uuid, glob, copy, re
from os import path
from threading import Thread

instrumentdb = JsonStore('storage/instrumentprofiles.json')
obsrundb = JsonStore('storage/observingruns.json')
linelistdb = JsonStore('storage/linelists.json')

def get_tracedir(inst):
    return instrumentdb.get(inst).tracedir
    #idb = shelve.open(instrumentdb)
    #out = idb[inst].tracedir
    #idb.close()
    #return out
    
def update_ntuple(ntuple, field, newval):
    t = type(ntuple)
    tmp = ntuple._asdict()
    for i, f in enumerate(field):
        tmp[f] = newval[i]
    return t(**tmp)

class BorderBox(BoxLayout):
    borderweight = NumericProperty(2)
    the_container = ObjectProperty(None)
    
    def add_widget(self, widget):
        if self.the_container is None:
            return super(BorderBox, self).add_widget(widget)
        return self.the_container.add_widget(widget)
    
    def remove_widget(self, widget):
        if self.the_container:
            return self.the_container.remove_widget(widget)
        return super(BorderBox, self).remove_widget(widget)

class ObsfileInsert(BoxLayout):
    obsfile = ObjectProperty(None)
    dithertype = StringProperty('')

    def launch_header(self):
        self.header_viewer = FitsHeaderDialog(fitsimage = self.obsfile)
        self.header_viewer.bind(on_dismiss = self.update_header())
        self.header_viewer.open()
    
    def update_header(self):
        self.obsfile.header = self.header_viewer.fitsimage.header

class SpecscrollInsert(BoxLayout):
    active = BooleanProperty(True)
    text = StringProperty('')
    spectrum = ObjectProperty(MeshLinePlot())

class ApertureSlider(BoxLayout):
    aperture_line = ObjectProperty(None)
    slider = ObjectProperty(None)
    trash = ObjectProperty(None)
    plot_points = ListProperty([])
    tfscreen = ObjectProperty(None)
    
    def fix_line(self, val):
        top_y = interp_x(self.plot_points, val)
        self.aperture_line.points = [(val, 0), (val, top_y)]

class IRScreen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(IRScreen, self).add_widget(*args)

#blank_instrument = {'direction':'horizontal', 'dimensions':(1024, 1024), \
#    'header':{'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, \
#    'description':'Instrument Profile Description'} 

class InstrumentScreen(IRScreen):
    saved_instrument_names = ListProperty([])
    saved_instruments = ListProperty([])
    instrument_list = ListProperty([])
    current_text = StringProperty('')
    current_instrument = ObjectProperty(ird.InstrumentProfile())
    trace_direction = StringProperty('horizontal')
    
    def on_pre_enter(self):
        #idb = shelve.open(instrumentdb)
        idb = instrumentdb
        self.saved_instrument_names = sorted(idb.keys())
        self.saved_instruments = [ird.InstrumentProfile(**idb[s]) for s in self.saved_instrument_names]
        #idb.close()
        self.instrument_list = [Button(text = x, size_hint_y = None, height = '30dp') \
            for x in self.saved_instrument_names]
    
    def set_instrument(self):
        self.current_text = self.ids.iprof.text
        try:
            ind = self.saved_instrument_names.index(self.current_text)
            self.current_instrument = self.saved_instruments[ind]
        except ValueError:
            self.current_instrument = ird.InstrumentProfile(instid=self.current_text)
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
        new_instrument = ird.InstrumentProfile(**args)
        self.current_instrument = new_instrument
        #idb = shelve.open(instrumentdb)
        #idb[new_instrument.instid] = new_instrument
        #idb.close()
        instrumentdb[new_instrument.instid] = new_instrument._asdict()
        self.on_pre_enter()
    
class ObservingScreen(IRScreen):
    obsids = DictProperty({})
    obsrun_list = ListProperty([])
    obsrun_buttons = ListProperty([])
    current_obsrun = ObjectProperty(ird.ObsRun())
    obsnight_list = ListProperty([])
    obsnight_buttons = ListProperty([])
    current_obsnight = ObjectProperty(ird.ObsNight())
    instrument_list = ListProperty([])
    caltype = StringProperty('')
    target_list = ListProperty([])
    current_target = ObjectProperty(ird.ObsTarget())
    file_list = ListProperty([])
    
    def __init__(self, **kwargs):
        super(ObservingScreen, self).__init__(**kwargs)
        self.waiting = WaitingDialog(text='Please wait while the calibration images build, thank you!')
    
    def on_enter(self):
        #idb = shelve.open(instrumentdb)
        #self.instrument_list = idb.keys()
        #idb.close()
        self.instrument_list = instrumentdb.keys()
        #odb = shelve.open(obsrundb)
        odb = obsrundb
        self.obsids = {x:odb[x][x] for x in odb}
        self.obsrun_list = odb.keys()
        #odb.close()
        self.obsrun_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsrun_list]
    
    def on_pre_leave(self):
        theapp = App.get_running_app()
        pairs = pair_dithers(self.current_target.dither)
        tmp = [(self.current_target.images[x] for x in y) for y in pairs]
        theapp.extract_pairs = [[self.current_target.images[x] for x in y] for y in pairs]
        theapp.current_target = self.current_target
        theapp.current_paths = {'cal':self.current_obsnight.calpath, 
            'raw':self.current_obsnight.rawpath, 'out':self.current_obsnight.outpath}
        theapp.current_night = self.current_obsnight
    
    def set_obsrun(self):
        run_id = self.ids.obsrun.text
        if run_id not in self.obsids:
            while True:
                run_db = 'storage/'+str(uuid.uuid4())+'.json'
                if not glob.glob(run_db+'*'):
                    break
            self.obsids[run_id] = run_db
            #odb = shelve.open(obsrundb)
            obsrundb[run_id] = {run_id:run_db}
            #odb.close()
        else:
            run_db = self.obsids[run_id]
        self.current_obsrun = ird.ObsRun(runid=run_id)
        self.rdb = JsonStore(run_db)
        #rdb = shelve.open(run_db)
        for r in self.rdb:
            ird.add_to(self.current_obsrun, ird.ObsNight(**self.rdb[r]))
        #rdb.close()
        self.obsnight_list = self.current_obsrun.nights.keys()
        self.obsnight_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsnight_list]
    
    def set_obsnight(self):
        night_id = self.ids.obsnight.text
        if night_id == '' or self.current_obsrun.runid == '':
            return
        if night_id not in self.obsnight_list:
            self.obsnight_list.append(night_id)
            self.obsnight_buttons.append(Button(text = night_id, \
                size_hint_y = None, height = 30))
            self.current_obsnight = ird.ObsNight(date = night_id)
            #self.current_obsrun.addnight(self.current_obsnight)
            ird.add_to(self.current_obsrun, self.current_obsnight)
        else:
            #self.current_obsnight = self.current_obsrun.get_night(night_id)
            ird.get_from(self.current_obsnight, night_id)
            self.ids.rawpath.text = self.current_obsnight.rawpath
            self.ids.outpath.text = self.current_obsnight.outpath
            self.ids.calpath.text = self.current_obsnight.calpath
            self.ids.fformat.text = self.current_obsnight.filestub
            self.set_filelist()
        #rdb = shelve.open(self.obsids[self.current_obsrun.runid])
        #rdb = JsonStore(self.obsids[self.current_obsrun.runid])
        for night in self.obsnight_list:
            self.rdb[night] = ird.get_from(self.current_obsrun, night)._asdict()
        #rdb.close()
        self.target_list = self.current_obsnight.targets.keys()
    
    def pick_rawpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('raw',popup.chosen_directory))
        popup.open()
    
    def setpath(self, which, dir):
        self.current_obsnight = update_ntuple(self.current_obsnight, [which+'path',], [dir,])
        if which == 'raw':
            self.ids.rawpath.text = dir
        elif which == 'out':
            self.ids.outpath.text = dir
        elif which == 'cal':
            self.ids.calpath.text = dir
        
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
            popup = WarningDialog(text = "File format is not valid; must use '#' as placeholder only")
            popup.open()
            return
        self.current_obsnight = update_ntuple(self.current_obsnight, ['filestub',], [stub,]) 
    
    def set_caltype(self, caltype):
        if caltype == 'Flats (lamps ON)':
            flist = self.current_obsnight.flaton.flist if self.current_obsnight.flaton else ''
        elif caltype == 'Flats (lamps OFF)':
            flist = self.current_obsnight.flatoff.flist if self.current_obsnight.flatoff else ''
        elif caltype == 'Arc Lamps':
            flist = self.current_obsnight.cals.flist if self.current_obsnight.cals else ''
        self.ids.calfiles.text = flist
        
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
        target = image_stack(flist, path.join(raw, stub), output = path.join(cal, outp))
        self.waiting.dismiss()
    
    def save_night(self):
        tmp = self.current_obsrun.nights
        tmp[self.current_obsnight.date] = self.current_obsnight
        update_ntuple(self.current_obsrun
        ird.add_to(self.current_obsrun, self.current_obsnight)
        #rdb = shelve.open(self.obsids[self.current_obsrun.runid])
        #rdb = JsonStore(self.obsids[self.current_obsrun.runid])
        for night in self.obsnight_list:
            print ird.get_from(self.current_obsrun, night)._asdict()
            self.rdb[night] = ird.get_from(self.current_obsrun, night)._asdict()
        print self.rdb._is_changed, self.rdb._data
        self.rdb.store_sync()
        #rdb.close()
        
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
        self.current_target = ird.ObsTarget(night=self.current_obsnight, **targs)
        ird.add_to(self.current_obsnight, self.current_target)
        self.target_list = self.current_obsnight.targets.keys()
        self.ids.targs.text = self.current_target.targid
        self.set_filelist()
        #rdb = shelve.open(self.obsids[self.current_obsrun.runid])
        #rdb = JsonStore(self.obsids[self.current_obsrun.runid])
        self.rdb[self.current_obsnight.date] = self.current_obsnight._asdict()
        #rdb.close()
    
    def set_filelist(self):
        self.ids.obsfiles.clear_widgets()
        self.file_list = []
        for file, dither in zip(self.current_target.images, self.current_target.dither):
            tmp = ObsfileInsert(obsfile = file, dithertype = dither)
            self.file_list.append(tmp)
            self.ids.obsfiles.add_widget(tmp)
    
    def save_target(self):
        self.current_target = update_ntuple(self.current_target, ['dither', 'notes'], \
            [[x.dithertype for x in self.file_list], self.ids.tnotes.text])
        #just make sure everything is propagating correctly
        self.current_obsnight.targets[self.current_target.targid] = self.current_target
        self.current_obsrun.nights[self.current_obsnight.date] = self.current_obsnight
        #rdb = shelve.open(self.obsids[self.current_obsrun.runid])
        #rdb = JsonStore(self.obsids[self.current_obsrun.runid])
        for night in self.obsnight_list:
            self.rdb[night] = ird.get_from(self.current_obsrun, night)._asdict
        #rdb.close()
        self.target_list = self.current_obsnight.targets.keys()

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
    current_impair = ObjectProperty(None)
    current_flats = ObjectProperty(None)
    theapp = ObjectProperty(None)
    
    def __init__(self):
        super(ExtractRegionScreen, self).__init__()
        self.ids.ipane.load_data(default_image)
        with self.imcanvas.canvas.after:
            c = Color(30./255., 227./255., 224./255.)
            self.regionline = Line(points=self.lcoords, close=True, \
                dash_length = 2, dash_offset = 1)
    
    def on_enter(self):
        flat = path.join(self.paths['cal'],'Flat.fits')
        if self.theapp.current_night.flaton:
            if self.theapp.current_night.flatoff:
                im_subtract(self.theapp.current_night.flaton, \
                    self.theapp.current_night.flatoff, outfile = flat)
            else:
                write_fits(flat, self.theapp.current_night.flaton.header, \
                    self.theapp.current_night.flaton.data_array)
            self.current_flats = FitsImage(flat)
            self.current_flats.load()
        self.pairstrings = ['{0} - {1}'.format(*[path.basename(x.fitsfile) for x in y]) for y in self.extract_pairs]
    
    def on_pre_leave(self):
        self.theapp.current_impair = self.current_impair
        self.theapp.current_flats = self.current_flats
    
    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = WarningDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',val)+'.fits'
        if not path.isfile(fitsfile):
            im1, im2 = [x for x in copy.deepcopy(self.extract_pairs[pair_index])]
            im1.load(); im2.load()
            if self.current_flats:
                im1 = im_divide(im1, self.current_flats)
                im2 = im_divide(im2, self.current_flats)
            im_subtract(im1, im2, outputfile=path.join(self.paths['out'],fitsfile))
        self.current_impair = FitsImage(path.join(self.paths['out'],fitsfile))
        self.current_impair.load()
        self.ids.ipane.load_data(self.current_impair)
        self.imwid, self.imht = self.current_impair.dimensions
        if self.current_impair.get_header_keyword('EXREGX1'):
            for x in ['x1', 'y1', 'x2', 'y2']:
                tmp = self.current_impair.get_header_keyword('EXREG'+x.upper())
                self.set_coord(x, tmp[0])
    
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
        if coord == 'x1':
            self.bx1 = value
        elif coord == 'x2':
            self.bx2 = value
        elif coord == 'y1':
            self.by1 = value
        elif coord == 'y2':
            self.by2 = value
        self.regionline.points = self.lcoords
    
    def save_region(self):
        self.current_impair.header['EXREGX1'] = (self.bx1, 'extraction region coordinate X1')
        self.current_impair.header['EXREGY1'] = (self.by1, 'extraction region coordinate Y1')
        self.current_impair.header['EXREGX2'] = (self.bx2, 'extraction region coordinate X2')
        self.current_impair.header['EXREGY2'] = (self.by2, 'extraction region coordinate Y2')
        self.current_impair.update_fits()

class TracefitScreen(IRScreen):
    paths = DictProperty([])
    itexture = ObjectProperty(Texture.create(size = (2048, 2048)))
    iregion = ObjectProperty(None)
    current_impair = ObjectProperty(None)
    extractregion = ObjectProperty(None)
    tplot = ObjectProperty(MeshLinePlot(color=[1,1,1,1])) #change to SmoothLinePlot when kivy 1.8.1
    current_target = ObjectProperty(None)
    pairstrings = ListProperty([])
    apertures = DictProperty({'pos':[], 'neg':[]})
    drange = ListProperty([0,1024])
    tracepoints = ListProperty([])
    trace_axis = NumericProperty(0)
    fit_params = DictProperty({})
    trace_lines = ListProperty([MeshLinePlot(color=[0,0,1,0]),MeshLinePlot(color=[0,1,1,0])])
    current_flats = ObjectProperty(None)
    theapp = ObjectProperty(None)
    
    def on_enter(self):
        self.pairstrings = ['{0} - {1}'.format(*[path.basename(x.fitsfile) for x in y]) \
            for y in theapp.extract_pairs]
            
    def set_imagepair(self, val):
        if not theapp.current_target:
            popup = WarningDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        self.pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',val)+'.fits'
        if not path.isfile(fitsfile):
            popup = WarningDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair = FitsImage(fitsfile)
        self.region = self.current_impair.get_header_keyword(*('EXREG' + x for x in ['X1','Y1','X2','Y2']))
        if not any(self.region):
            popup = WarningDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair.load()
        idata = ''.join(map(chr,self.current_impair.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.current_impair.dimensions)
        self.trace_axis = 0 if get_tracedir(self.current_target.instrument_id) == 'vertical' else 1
        reg = self.region
        tmp = self.current_impair.data_array[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
        self.extractregion = med_normal(tmp)
        if not self.trace_axis:
            self.extractregion = self.extractregion.transpose()
            self.trace_axis = 1
            self.region = [self.region[x] for x in [1, 0, 3, 2]]
        reg = self.region[:]
        reg[2] = reg[2] - reg[0]
        reg[3] = reg[3] - reg[1]
        self.iregion = self.itexture.get_region(*reg)
        dims = [[0,0],list(self.extractregion.shape)]
        dims[0][self.trace_axis] = 0.4 * self.extractregion.shape[self.trace_axis]
        dims[1][self.trace_axis] = 0.6 * self.extractregion.shape[self.trace_axis]
        self.tracepoints = robm(self.extractregion[dims[0][0]:dims[1][0]+1,\
            dims[0][1]:dims[1][1]+1], axis = self.trace_axis)
        self.tplot.points = zip(range(len(self.tracepoints)), self.tracepoints)
        self.drange = minmax(self.tracepoints)
        self.ids.the_graph.add_plot(self.tplot)
    
    def add_postrace(self):
        peaks = find_peaks(self.tracepoints, len(self.apertures['pos'])+1, \
            tracedir = self.trace_axis)
        #new_peak = float(peaks[-1])
        new_peak = float(peaks)
        peakheight = interp_x(self.tplot.points, new_peak)
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
        peaks = find_peaks(self.tracepoints, len(self.apertures['neg'])+1, \
            tracedir = self.trace_axis, pn='neg')
        #new_peak = float(peaks[-1])
        new_peak = float(peaks)
        peakheight = interp_x(self.tplot.points, new_peak)
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
        popup = SetFitParams()
        popup.bind(on_dismiss = lambda x: self.setfp(popup.fit_args))
        popup.open()
    
    def setfp(self, args):
        self.fit_params = args
        
    def fit_trace(self):
        if not self.fit_params or self.fit_params['shape'] not in ('Gaussian','Lorentzian'):
            popup = WarningDialog(text='Make sure you set up your fit parameters!')
            popup.open()
            return
        pos = {'pos':[x.slider.value for x in self.apertures['pos']], \
            'neg':[x.slider.value for x in self.apertures['neg']]}
        for x in self.trace_lines:
            if x in self.ids.the_graph.plots:
                self.the_graph.remove_plot(x)
        if self.fit_params.get('man',False):
            popup = DefineTrace(npos=len(self.apertures['pos']), \
                nneg=len(self.apertures['neg']), imtexture = self.iregion)
            popup.bind(on_dismiss = self.manual_trace(popup.tracepoints))
            popup.open()
            return
        self.xx, self.fitparams['pmodel'], self.fitparams['nmodel'] = \
            fit_multipeak(self.tracepoints, pos = pos, wid = self.fit_params['wid'], \
            ptype = self.fit_params['shape'])
        self.trace_lines[0].points = zip(self.xx, self.fitparams['pmodel'](self.xx))
        self.trace_lines[1].points = zip(self.xx, self.fitparams['nmodel'](self.yy))
        self.ids.the_graph.add_plot(self.trace_lines[0])
        self.ids.the_graph.add_plot(self.trace_lines[1])
        
    def fix_distort(self):
        if not (self.fit_params.get('pmodel',False) or \
            self.fit_params.get('nmodel',False)):
            popup = WarningDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        pdistort, ndistort = draw_trace(self.extractregion, self.xx, self.fitparams['pmodel'], \
            self.fitparams['nmodel'], fixdistort = True, fitdegree = self.fitparams['deg'])
        
        im1, im2 = [x for x in copy.deepcopy(theapp.extract_pairs[self.pair_index])]
        im1.load(); im2.load()
        if self.current_flats:
            im1 = im_divide(im1, self.current_flats)
            im2 = im_divide(im2, self.current_flats)
        im1.data_array = undistort_imagearray(im1.data_array, pdistort)
        im2.data_array = undistort_imagearray(im2.data_array, ndistort)
        im_subtract(im1, im2, outputfile=self.current_impair.fitsfile)
        tmp = self.current_impair
        self.current_impair = FitsImage(self.current_impair.fitsfile)
        self.current_impair.header['EXREGX1'] = (tmp.get_header_keyword('EXREGX1'), 'extraction region coordinate X1')
        self.current_impair.header['EXREGY1'] = (tmp.get_header_keyword('EXREGY1'), 'extraction region coordinate Y1')
        self.current_impair.header['EXREGX2'] = (tmp.get_header_keyword('EXREGX2'), 'extraction region coordinate X2')
        self.current_impair.header['EXREGY2'] = (tmp.get_header_keyword('EXREGY2'), 'extraction region coordinate Y2')
        self.current_impair.update_fits()
        self.set_imagepair(self.pairstrings[self.pair_index])
        self.fit_params['nmodel'] = None
        self.fit_params['pmodel'] = None
        
    
    def manual_trace(self, traces):
        pass #need to figure out how to apply these
    
    def extract_spectrum(self):
        if not (self.fit_params.get('pmodel',False) or \
            self.fit_params.get('nmodel',False)):
            popup = WarningDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        
        #need a calibration, too
        self.lamp = None
        if theapp.current_night.cals:
            self.lamp = theapp.current_night.cals.data_array if not self.current_flats \
                else im_divide(theapp.current_night.cals, self.current_flats).data_array
            self.lamp = self.lamp[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
        im1, im2 = [x for x in copy.deepcopy(the_app.extract_pairs[self.pair_index])]
        im1.load(); im2.load()
        if self.current_flats:
            im1 = im_divide(im1, self.current_flats)
            im2 = im_divide(im2, self.current_flats)
        im1.data_array = undistort_imagearray(im1.data_array, pdistort)
        im2.data_array = undistort_imagearray(im2.data_array, ndistort)
        tmp, self.tell = im_minimum(im1.data_array, im2.data_array)
        self.tell = self.tell[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
        self.pextract = extract(self.fit_params['pmodel'], self.extractregion, self.tell, 'pos', lamp = self.lamp)
        self.nextract = extract(self.fit_params['nmodel'], self.extractregion, self.tell, 'neg', lamp = self.lamp)
        
        #write uncalibrated spectra to fits files (will update after calibration)
        pstub = self.paths['out'] + re.sub('.fits','-ap%i',im1.fitsfile)
        ext = ('.fits','-sky.fits','-lamp.fits')
        h = im1.header
        for i, p_ap in enumerate(self.pextract):
            for j in range(p_ap.shape[1]):
                spec = p_ap[:,j]
                write_fits((pstub + ext[i]) % j, h, spec)
        
        nstub = self.paths['out'] + re.sub('.fits','-ap%i',im2.fitsfile)
        h = im2.header
        for i, n_ap in enumerate(self.nextract):
            for j in range(n_ap.shape[1]):
                spec = n_ap[:,j]
                write_fits((nstub + ext[i]) % j, h, spec)
        
        
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
        #lldb = shelve.open(linelistdb)
        self.linelists = linelistdb.keys()
        self.linelist_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in linelistdb]
        #lldb.close()
    
    def set_spectrum(self, spec):
        self.spec_index = self.speclist.index(spec)
        try:
            tmp = ExtractedSpectrum(self.paths['out']+self.speclist[self.spec_index] + '.fits')
        except:
            popup = WarningDialog(text="You haven't extracted that spectrum yet!")
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
            popup = WarningDialog(text="Please select a line list first.")
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
            popup = WarningDialog(text="You don't have a calibration of this type...")
            popup.open()
            return
        niter = self.ids.numiter.text
        if self.linelist in self.linelists:
            #lldb = shelve.open('storage/linelists')
            linelist_path = linelistdb[self.lineslist]
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
    
    def on_enter(self):
        flist = [x for x in glob.iglob(self.paths['out'] + y + '-ap*.fits') for y in self.speclist]
        self.the_specs = [ExtractedSpectrum(x) for x in flist]
        colors = gen_colors(len(self.the_specs))
        for i, ts in enumerate(self.the_specs):
            tmp = SpecscrollInsert(text=flist[i])
            tmp.spectrum.color = colors[i] + (1)
            tmp.bind(active=toggle_spectrum(i))
            if not ts.wav:
                tmp.active = False
                self.scaled_spectra.append([xrange(len(ts.spec)),ts.spec])
                tmp.spectrum.points = zip(*self.scaled_spectra[i])
                self.spec_inserts.append(tmp)
                continue
            self.scaled_spectra.append([ts.wav,ts.spec])
            tmp.spectrum.points = zip(*self.scaled_spectra[i])
            tmp.bind(active=toggle_spectrum(i))
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
        ref = (self.the_specs[self.ind].wav, self.the_specs[self.ind].spec)
        for i, s in enumerate(self.the_specs):
            self.scaled_spectra[i] = [s.wav, s.spec] if i == self.ind else \
                scale_spec(ref, [s.wav, s.spec])
            self.spec_inserts[i].spectrum.points = zip(*self.scaled_spectra[i])
        self.comb_method.dispatch()
        self.setminmax()
    
    def on_comb_method(self, instance, value):
        specs = [x[1] for i, x in enumerate(self.scaled_spectra) if self.spec_inserts[i].active]
        comb = combine_spectra(specs, value)
        self.combined_spectrum.points = zip(self.scaled_spectra[self.ind][0], comb)
    
    def combine(self):
        out = self.ids.savefile.text
        h = self.the_specs[self.ind].header
        write_fits(out, h, zip(*self.combined_spectrum.points))

class TelluricScreen(IRScreen):
    pass

class IRReduceApp(App):
    current_title = StringProperty('')
    index = NumericProperty(-1)
    screen_names = ListProperty([])
    extract_pairs = ListProperty([])
    current_night = ObjectProperty(ird.ObsNight())
    current_target = ObjectProperty(ird.ObsTarget())
    current_paths = DictProperty({})
    current_impair = ObjectProperty(None)
    current_flats = ObjectProperty(None)
    
    def build(self):
        self.title = 'IR-Reduce'
        self.icon = 'resources/irreduc-icon.png'
        self.screen_names = ['Instrument Profile',
            'Observing Run', 'Extraction Region',
            'Trace Fitting', 'Wavelength Calibration',
            'Combine Spectra', 'Telluric Correction']
        self.shortnames = ['instrument','observing',
            'region','trace','wavecal','combine','telluric']
        sm = self.root.ids.sm
        sm.add_widget(InstrumentScreen())
        sm.add_widget(ObservingScreen())
        sm.add_widget(ExtractRegionScreen())
        sm.add_widget(TracefitScreen())
        sm.add_widget(WavecalScreen())
        sm.add_widget(CombineScreen())
        sm.add_widget(TelluricScreen())
        sm.current = 'instrument'
        self.index = 0
        self.current_title = self.screen_names[self.index]
    
    def on_pause(self):
        return True

    def on_resume(self):
        pass
    
    def on_current_title(self, instance, value):
        self.root.ids.spnr.text = value
    
    def go_previous_screen(self):
        self.index = (self.index - 1) % len(self.screen_names)
        self.update_screen()
    
    def go_next_screen(self):
        self.index = (self.index + 1) % len(self.screen_names)
        self.update_screen()
        
    def go_screen(self, idx):
        self.index = idx
        self.update_screen()
    
    def update_screen(self):
        self.root.ids.sm.current = self.shortnames[self.index]
        self.current_title = self.screen_names[self.index]

if __name__ == '__main__':
    IRReduceApp().run()