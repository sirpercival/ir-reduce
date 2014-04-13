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
    StringProperty, DictProperty, AliasProperty
from kivy.uix.widget import Widget
from kivy.uix.popup import Popip
from kivy.uix.label import Labels
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from imagepane import ImagePane, default_image
from fitsimage import FitsImage
from comboedit import ComboEdit
from ir_databases import InstrumentProfile, ObsRun, ObsTarget
from dialogs import FitsHeaderDialog, DirChooser, AddTarget, SetFitParams, WarningDialog
from imarith import pair_dithers, im_subtract
from robuststats import robust_mean as robm, interp_x, idlhash
from findtrace import find_peaks, fit_multipeak, draw_trace
import shelve, uuid, glob, copy, re
from os import path

instrumentdb = 'storage/instrumentprofiles'
obsrundb = 'storage/observingruns'

def get_tracedir(inst):
    idb = shelve.open(instrumentdb)
    out = idb[inst].tracedir
    idb.close()
    return out

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
        header_viewer = FitsHeaderDialog(fitsimage = obsfile)
        header_viewer.open()

class ApertureSlider(BoxLayout):
    aperture_line = ObjectProperty(None)
    slider = ObjectProperty(None)
    trash = ObjectProperty(None)
    plot_points = ListProperty([])
    
    def fix_line(self, val):
        top_y = interp_x(self.plot_points, val)
        self.aperture_line.points = [(val, 0), (val, top_y)]

class IRScreen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(IRScreen, self).add_widget(*args)

blank_instrument = {'direction':'horizontal', 'dimensions':(1024, 1024), \
    'header':{'exp':'EXPTIME', 'air':'AIRMASS', 'type':'IMAGETYP'}, \
    'description':'Instrument Profile Description'} 

class InstrumentScreen(IRScreen):
    saved_instrument_names = ListProperty([])
    saved_instruments = ListProperty([])
    instrument_list = ListProperty([])
    current_text = StringProperty('')
    current_instrument = ObjectProperty(InstrumentProfile(id='', blank_instrument))
    trace_direction = StringProperty('horizontal')
    
    def on_pre_enter(self):
        idb = shelve.open(instrumentdb)
        self.saved_instrument_names = sorted(idb.keys())
        self.saved_instruments = [idb[s] for s in self.saved_instruments]
        idb.close()
        self.instrument_list = [Button(text = x, size_hint_y = None, height = 30) \
            for x in self.saved_instrument_names]
    
    def set_instrument(self):
        self.current_text = self.ids.iprof.text
        if self.current_text in self.saved_instruments:
            self.current_instrument = self.instrument_list[self.current_text]
        else:
            self.current_instrument = InstrumentProfile(id=self.current_text, \
                blank_instrument)
    
    def save_instrument(self):
        new_instrument = InstrumentProfile(id = self.current_text, \
            direction = 'horizontal' if self.ids.trace_h.state == 'down' else 'vertical', \
            dimensions = (self.ids.xdim.text, self.ids.ydim.text), \
            description = self.ids.idesc.text, header = {'exp':self.ids.etime.text, \
                'air':self.ids.secz.text, 'type':self.ids.itype.text})
        self.current_instrument = new_instrument
        idb = shelve.open(instrumentdb)
        idb[new_instrument.id] = new_instrument
        idb.close()
        self.on_pre_enter()
        
blank_night = {'filestub':'', 'rawpath':'', 'outpath':'', 'calpath':'', \
    'flaton':None, 'flatoff':None, 'cals':None}
blank_target = {'files':'', 'iid':''}
    
class ObservingScreen(IRScreen):
    obsids = DictProperty({})
    obsrun_list = ListProperty([])
    obsrun_buttons = ListProperty([])
    current_obsrun = ObjectProperty(ObsRun(id=''))
    obsnight_list = ListProperty([])
    obsnight_buttons = ListProperty([])
    current_obsnight = ObjectProperty(ObsNight(date='', blank_night))
    instrument_list = ListProperty([])
    caltype = StringProperty('')
    target_list = ListProperty([])
    current_target = ObjectProperty(ObsTarget(id='', blank_target))
    file_list = ListProperty([])
    
    def on_enter(self):
        idb = shelve.open(instrumentdb)
        self.instrument_list = idb.keys()
        idb.close()
        odb = shelve.open(obsrundb)
        self.obsids = {x:odb[x] for x in odb}
        odb.close()
        self.obsrun_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsrun_list]
    
    def on_pre_leave(self):
        theapp = App.get_running_app()
        pairs = pair_dithers(self.current_target.dither)
        theapp.extract_pairs = [(self.current_target.images[x] for x in y) for y in pairs]
        theapp.current_target = self.current_target
        theapp.current_paths = {'cal':self.current_obsnight.calpath, 
            'raw':self.current_obsnight.rawpath, 'out':self.current_obsnight.outpath}
    
    def set_obsrun(self):
        run_id = self.ids.obsrun.text
        if run_id not in self.obsids:
            while True
                run_db = 'storage/'+uuid.uuid4()
                if not glob.glob(run_db+'*'):
                    break
            self.obsids[run_id] = run_db
            odb = shelve.open(obsrundb)
            odb[run_id] = run_db
            odb.close()
        else:
            run_db = self.obsids[run_id]
        self.current_obsrun = ObsRun(id=run_id)
        rdb = shelve.open(run_db)
        for r in rdb:
            self.current_obsrun.addnight(rdb[r])
        rdb.close()
        self.obsnight_list = self.current_obsrun.nights.keys()
        self.obsnight_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsnight_list]
    
    def set_obsnight(self):
        night_id = self.ids.obsnight.text
        if night_id not in self.obsnight_list:
            self.obsnight_list.append(night_id)
            self.obsnight_buttons.append(Button(text = night_id, \
                size_hint_y = None, height = 30))
            self.current_obsnight = ObsNight(date = night_id, blank_night)
            self.current_obsrun.addnight(self.current_obsnight)
        else:
            self.current_obsnight = self.current_obsrun.get_night(night_id)
        rdb = shelve.open(self.obsids[self.current_obsrun.id])
        for night in self.obsnight_list:
            rdb[night] = self.current_obsrun.get_night(night)
        rdb.close()
        self.target_list = self.current_obsnight.targets.keys()
    
    def pick_rawpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.current_obsnight.rawpath = popup.chosen_directory)
        popup.open()
        
    def pick_outpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.current_obsnight.outpath = popup.chosen_directory)
        popup.open()
        
    def pick_calpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.current_obsnight.calpath = popup.chosen_directory)
        popup.open()
    
    def set_caltype(self, caltype):
        if caltype == 'Flats (lamps ON)':
            flist = self.current_obsnight.flaton.flist if self.current_obsnight.flaton else ''
        elif caltype == 'Flats (lamps OFF)':
            flist = self.current_obsnight.flatoff.flist if self.current_obsnight.flatoff else ''
        elif caltype == 'Arc Lamps':
            flist = self.current_obsnight.cals.flist if self.current_obsnight.cals else ''
        self.ids.calfiles.text = flist
        
    def make_cals(self):
        caltype = self.ids.caltypes.text
        flist = self.ids.calfiles.text
        if caltype == 'Flats (lamps ON)':
            self.current_obsnight.flaton = self.current_obsnight.add_stack('flats-on', flist, \
                output=self.current_obsnight.date+'-FlatON.fits')
        elif caltype == 'Flats (lamps OFF)':
            self.current_obsnight.flatoff = self.current_obsnight.add_stack('flats-off', flist, \
                output=self.current_obsnight.date+'-FlatOFF.fits')
        elif caltype == 'Arc Lamps':
            self.current_obsnight.cals = self.current_obsnight.add_stack('cals', flist, \
                output=self.current_obsnight.date+'-Wavecal.fits')
    
    def save_night(self):
        self.current_obsrun[self.current_obsnight.date] = self.current_obsnight
        rdb = shelve.open(self.obsids[self.current_obsrun.id])
        for night in self.obsnight_list:
            rdb[night] = self.current_obsrun.get_night(night)
        rdb.close()
        
    def set_target(self):
        target_id = self.ids.targs.text
        self.current_target = self.current_obsnight.targets[target_id]
        self.set_filelist()
    
    def add_target(self):
        popup = AddTarget(instrumentlist = self.instrument_list)
        popup.bind(on_dismiss: self.update_targets(popup.target_args))
        popup.open()
    
    def update_targets(self, targs):
        self.current_target = ObsTarget(targs)
        self.current_obsnight.add_target(self.current_target)
        self.target_list = self.current_obsnight.targets.keys()
        self.ids.targs.text = self.current_target.id
        self.set_filelist()
        rdb = shelve.open(self.obsids[self.current_obsrun.id])
        rdb[self.current_obsnight.date] = self.current_obsnight
        rdb.close()
    
    def set_filelist(self):
        self.ids.obsfiles.clear_widgets()
        self.file_list = []
        for file, dither in zip(self.current_target.images, self.current_target.dither):
            tmp = ObsfileInsert(obsfile = file, dithertype = dither)
            self.file_list.append(tmp)
            self.ids.obsfiles.add_widget(tmp)
    
    def save_target(self):
        self.current_target.dither = [x.dithertype for x in self.file_list]
        self.current_target.notes = self.ids.tnotes.text
        #just make sure everything is propagating correctly
        self.current_obsnight.targets[self.current_target.id] = self.current_target
        self.current_obsrun.nights[self.current_obsnight.date] = self.current_obsnight
        rdb = shelve.open(self.obsids[self.current_obsrun.id])
        for night in self.obsnight_list:
            rdb[night] = self.current_obsrun.get_night(night)
        rdb.close()
        self.target_list = self.current_obsnight.targets.keys()

class ExtractRegionScreen(IRScreen):
    paths = DictProperty({})
    extract_pairs = ListProperty([])
    pairstrings = ListProperty([])
    imwid = NumericProperty(1024)
    imht = NumericProperty(1024)
    bx1 = NumericProperty(0)
    bx2 = NumericProperty(self.imwid)
    by1 = NumericProperty(0)
    by2 = NumericProperty(self.imht)
    the_image = ObjectProperty(None)
    current_impair = ObjectProperty(None)
    
    def __init__(self):
        super(ExtractRegionScreen, self).__init__()
        self.ids.ipane.load_data(default_image)
        with self.the_image.canvas.after:
            c = Color(30./255., 227./255., 224./255.)
            self.regionline = Line(points=self.lcoords, close=True, \
                dash_length = 2, dash_offset = 1)
    
    def on_pre_leave(self):
        theapp = App.get_running_app()
        theapp.current_impair = self.current_impair
    
    def set_imagepair(self, val):
        pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',val)+'.fits'
        if not path.isfile(fitsfile):
            im1, im2 = [x.load() for x in copy.deepcopy(self.extract_pairs[pair_index])]
            im_subtract(im1, im2, outfile=fitsfile)
        self.current_impair = FitsImage(fitsfile)
        self.current_impair.load()
        self.ids.ipane.load_data(self.current_impair)
        self.imwid, self.imht = self.current_impair.dimensions
    
    def get_coords(self):
        xscale = float(self.the_image.width) / float(self.imwid)
        yscale = float(self.the_image.height) / float(self.imht)
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
        elif coord = 'y2':
            self.by2 = value
        self.regionline.points = self.lcoords
    
    def save_region(self):
        self.current_impair.header['EXREGX1'] = (self.bx1, 'extraction region coordinate X1')
        self.current_impair.header['EXREGY1'] = (self.by1, 'extraction region coordinate Y1')
        self.current_impair.header['EXREGX2'] = (self.bx2, 'extraction region coordinate X2')
        self.current_impair.header['EXREGY2'] = (self.by2, 'extraction region coordinate Y2')
        self.current_impair.update_fits()

class TracefitScreen(IRScreen):
    paths = ListProperty([])
    itexture = ObjectProperty(Texture.create(size = (2048, 2048)))
    iregion = ObjectProperty(None)
    current_impair = ObjectProperty(None)
    extractregion = ObjectProperty(None)
    tplot = ObjectProperty(MeshLinePlot(color=[1,1,1,1])) #change to SmoothLinePlot when kivy 1.8.1
    current_target = ObjectProperty(None)
    pairstrings = ListProperty([])
    apertures = DictProperty({'pos':[], 'neg':[]})
    drange = ListProperty([])
    tracepoints = ListProperty([])
    trace_axis = NumericProperty(0)
    fit_params = DictProperty({})
    trace_line = MeshLinePlot(color=[0,0,1,0])
    
    def set_imagepair(self, val):
        pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',val)+'.fits'
        if not path.isfile(fitsfile):
            popup = WarningDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair = FitsImage(fitsfile)
        region = self.current_impair.get_header_keyword(['EXREG' + x for x in ['X1','Y1','X2','Y2']])
        if not any(region):
            popup = WarningDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair.load()
        idata = ''.join(map(chr,self.current_impair.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.current_impair.dimensions)
        self.trace_axis = 0 if get_tracedir(self.current_target.instrument_id) == 'vertical' else 1
        self.extractregion = self.current_impair.data_array[region[1]:region[3]+1,region[0]:region[2]+1]
        if not self.trace_axis:
            self.extractregion = self.extractregion.transpose()
            self.trace_axis = 1
            region = [region[x] for x in [1, 0, 3, 2]]
        region[2] = region[2] - region[0]
        region[3] = region[3] - region[1]
        self.iregion = self.itexture.get_region(*region)
        dims = idlhash(self.extractregion.shape,[0.4,0.6], list=True)
        self.tracepoints = robm(self.extractregion[dims[0],dims[1]], axis = self.trace_axis)
        self.tplot.points = zip(range(len(tracepoints.tolist())), tracepoints.tolist())
        self.drange = [nanmin(self.tracepoints), nanmax(self.tracepoints)]
        self.ids.the_graph.add_plot(self.tplot)
    
    def add_postrace(self):
        peaks = find_peaks(self.tracepoints, len(self.apertures['pos'])+1, \
            tracedir = self.trace_axis)
        new_peak = peaks[-1]
        plot = MeshLinePlot(color=[0,1,0,1], \
            points=[(new_peak, 0), (new_peak, self.tracepoints[new_peak])])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = new_peak
        newspin.button.bind(on_press = self.remtrace('pos',newspin))
        self.ids.postrace.add_widget(newspin)
        self.apertures['pos'].append(newspin)
        
    def add_negtrace(self):
        peaks = find_peaks(self.tracepoints, len(self.apertures['neg'])+1, \
            tracedir = self.trace_axis, pn='neg')
        new_peak = peaks[-1]
        plot = MeshLinePlot(color=[0,1,0,1], \
            points=[(new_peak, 0), (new_peak, self.tracepoints[new_peak])])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = new_peak
        newspin.button.bind(on_press = self.remtrace('neg',newspin))
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
        popup.bind(on_dismiss: self.fit_params = popup.fit_args)
        popup.open()
        
    def fit_trace(self):
        if not self.fit_params:
            popup = WarningDialog(text='Make sure you set up your fit parameters!')
            popup.open()
            return
        pos = [x.slider.value for x in self.apertures['pos']] + \
            [x.slider.value for x in self.apertures['neg']]
        if self.trace_line in self.the_graph.plots:
            self.the_graph.remove_plot(self.trace_line)
        self.xx, self.fitparams['model'] = fit_multipeak(self.tracepoints, npeak = len(pos), \
            pos = pos, wid = self.fit_params['wid'], ptype = self.fit_params['shape'])
        self.trace_line.points = zip(self.xx, self.fmodel(xx))
        self.the_graph.add_plot(self.trace_line)
        
    def fix_distort(self):
        if not self.fit_params.get('model',False):
            popup = WarningDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        draw_trace(self.xx, 
    
    def extract_spectrum(self):
        pass
        

class IRReduceApp(App):
    current_title = StringProperty('')
    index = NumericProperty(-1)
    screen_names = ListProperty([])
    extract_pairs = ListProperty([])
    current_target = ObjectProperty(None)
    current_paths = DictProperty({})
    current_impair = ObjectProperty(None)
    
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
        sm.current_title = self.screen_names[self.index]
    
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