# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:44:27 2015

@author: gray
"""

from irscreen import IRScreen
from kivy.lang import Builder
from kivy.properties import (DictProperty, ListProperty, NumericProperty,
                             ObjectProperty)
from kivy.garden.graph import SmoothLinePlot
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
import os, re, copy
from ..datatypes import FitsImage, RobustData
from utils.image_arithmetic import im_subtract, im_minimum, im_divide
from ..dialogs import AlertDialog, SetFitParams, DefineTrace
from astropy.io import fits
from ..persistence import tracedir
from ..extract import (fit_peaks, fit_trace as draw_trace, fix_distortion, 
                       extract, find_peak_1d)
import numpy as np

Builder.load_string('''
<ApertureSlider>:
    slider: sl
    trash: bu
    plot_points: root.tfscreen.tplot.points
    Slider:
        id: sl
        on_value: root.fix_line(self.value)
    AnchorLayout:
        size_hint: 0.25, 1
        Button:
            id: bu
            background_normal: 'atlas://resources/buttons/gc-normal'
            background_down: 'atlas://resources/buttons/gc-pressed'
            size_hint: None, None
            size: 24, 24

<TracefitScreen>:
    name: 'trace'
    id: tfscreen
    paths: app.current_paths
    current_target: app.current_target
    current_flats: app.current_flats
    theapp: app
    BoxLayout:
        orientation: 'vertical'
        size_hint_x: 0.4
        Spinner:
            text: 'Choose an image pair:'
            values: root.pairstrings
            on_text: root.set_imagepair(self.text)
            size_hint: 1, 0.15
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.15
            Label:
                text: 'Positive Apertures:'
            Button:
                background_normal: 'atlas://resources/buttons/an-normal'
                background_down: 'atlas://resources/buttons/an-pressed'
                size_hint: None, None
                size: 24, 24
                on_press: root.add_postrace()
        BorderBox:
            size_hint_y: 0.35
            ScrollView:
                do_scroll_x: False
                effect_cls: Factory.DampedScrollEffect
                GridLayout:
                    cols: 1
                    id: postrace
                    spacing: [0,3]
                    row_default_height: 24
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.15
            Label:
                text: 'Negative Apertures:'
            Button:
                background_normal: 'atlas://resources/buttons/an-normal'
                background_down: 'atlas://resources/buttons/an-pressed'
                size_hint: None, None
                size: 24, 24
                on_press: root.add_negtrace()
        BorderBox:
            size_hint_y: 0.35
            ScrollView:
                do_scroll_x: False
                effect_cls: Factory.DampedScrollEffect
                GridLayout:
                    cols: 1
                    id: negtrace
                    spacing: [0,3]
                    row_default_height: 24
        GridLayout:
            size_hint_y: 0.3
            cols: 2
            Button:
                text: 'Set Fit (1)'
                on_press: root.set_psf()
            Button:
                text: 'Fit Trace (2, 4)'
                on_press: root.fit_trace()
            Button:
                text: 'Fix Distortion (3)'
                on_press: root.fix_distort()
            Button:
                text: 'Extract Spectrum (5)'
                on_press: root.extract_spectrum()
        Widget:
    BoxLayout:
        orientation: 'vertical'
        size_hint_x: 0.6
        Graph:
            id: the_graph
            xlabel: 'Pixel'
            ylabel: 'Mean Counts'
            x_ticks_minor: 5
            x_ticks_major: 10
            y_ticks_minor: 5
            y_ticks_major: int((root.drange[1] - root.drange[0]) / 5.)
            x_grid_label: True
            y_grid_label: True
            xlog: False
            ylog: False
            x_grid: False
            y_grid: False
            xmin: 0
            xmax: len(root.tplot.points)
            ymin: root.drange[0]
            ymax: root.drange[1]
            label_options: {'bold': True}
        StencilView:
            size_hint_y: 0.7
            canvas.before:
                Rectangle:
                    size: self.size
                    pos: self.pos
                    texture: root.iregion
''')

def points_to_array(points):
    x, y = zip(*points)
    return np.array(y)

def make_region(im1, im2, region, flat = None, telluric = False):
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
    return RobustData(im_subtract(reg1, reg2)).normalized

class ApertureSlider(BoxLayout):
    aperture_line = ObjectProperty(None)
    slider = ObjectProperty(None)
    trash = ObjectProperty(None)
    plot_points = ListProperty([])
    tfscreen = ObjectProperty(None)
    
    def fix_line(self, val):
        x, y = zip(*self.plot_points)
        y = RobustData(y, x=x)
        top_y = y.interp(val)
        self.aperture_line.points = [(val, 0), (val, top_y)]

class TracefitScreen(IRScreen):
    paths = DictProperty([])
    itexture = ObjectProperty(Texture.create(size = (2048, 2048)))
    iregion = ObjectProperty(None)
    current_impair = ObjectProperty(None)
    extractregion = ObjectProperty(None)
    tplot = ObjectProperty(SmoothLinePlot(color=[1,1,1,1]))
    current_target = ObjectProperty(None)
    pairstrings = ListProperty([])
    apertures = DictProperty({'pos':[], 'neg':[]})
    drange = ListProperty([0,1024])
    tracepoints = ListProperty([])
    trace_axis = NumericProperty(0)
    fit_params = DictProperty({})
    trace_lines = ListProperty([SmoothLinePlot(color=[0,0,1,1]),SmoothLinePlot(color=[0,1,1,1])])
    current_flats = ObjectProperty(None)
    theapp = ObjectProperty(None)
    
    def on_enter(self):
        self.pairstrings = ['{0} - {1}'.format(*[os.path.basename(x.fitsfile) for x in y]) \
            for y in self.theapp.extract_pairs]
        if not self.theapp.current_flats and self.theapp.current_night.flaton:
            flat = os.path.join(self.paths['cal'],'Flat.fits')
            try:
                self.current_flats = FitsImage(flat, load = True)
                self.theapp.current_flats = self.current_flats
            except:
                if self.theapp.current_night.flaton and self.theapp.current_night.flaton[0]:
                    fon = FitsImage(os.path.join(self.paths['cal'], \
                        self.theapp.current_night.flaton[0]), load=True)
                    if self.theapp.current_night.flatoff and self.theapp.current_night.flatoff[0]:
                        foff = FitsImage(os.path.join(self.paths['cal'], \
                            self.theapp.current_night.flatoff[0]), load=True)
                        im_subtract(fon, foff, outputfile = flat)
                    else:
                        fits.writeto(flat, fon.data_array, header=fon.header)
                    self.current_flats = FitsImage(flat, load = True)
                self.theapp.current_flats = self.current_flats
        else:
            self.current_flats = self.theapp.current_flats

    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        self.pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',re.sub('.fits','',val))+'.fits'
        if not os.path.isfile(fitsfile):
            popup = AlertDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair = FitsImage(fitsfile)
        self.region = self.current_impair.get_header_keyword(*('EXREG' + x for x in ['X1','Y1','X2','Y2']))
        if not any(self.region):
            popup = AlertDialog(text='You have to select an extraction'\
                'region for this image pair \nbefore you can move on to this step.')
            popup.open()
            return
        self.current_impair.load()
        idata = ''.join(map(chr,self.current_impair.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.current_impair.dimensions)
        self.trace_axis = 0 if tracedir(self.current_target.instrument_id) == 'vertical' else 1
        tmp = self.theapp.extract_pairs[self.pair_index]
        if self.trace_axis:
            #self.trace_axis = 1
            reg = [self.region[x] for x in [1, 0, 3, 2]]
            print reg, tmp[0], tmp[1], self.current_flats
            self.extractregion = make_region(tmp[0], tmp[1], reg, self.current_flats)#.transpose()
        else:
            self.extractregion = make_region(tmp[0], tmp[1], self.region, self.current_flats).transpose()
        reg = self.region[:]
        reg[2] = reg[2] - reg[0]
        reg[3] = reg[3] - reg[1]
        self.iregion = self.itexture.get_region(*reg)
        dims = [[0,0],list(self.extractregion.shape)]
        dims[0][self.trace_axis] = 0.4 * self.extractregion.shape[self.trace_axis]
        dims[1][self.trace_axis] = 0.6 * self.extractregion.shape[self.trace_axis]
        self.tracepoints, dummy, dummy = RobustData(self.extractregion[dims[0][0]:dims[1][0]+1,\
            dims[0][1]:dims[1][1]+1], axis = self.trace_axis).stats()
        points = RobustData(self.tracepoints, x=list(range(len(self.tracepoints))))
        self.tplot.points = points.replace_nans()
        pxs, self.tracepoints = zip(*self.tplot.points)
        self.drange = RobustData(points_to_array(self.tracepoints)).minmax
        self.ids.the_graph.add_plot(self.tplot)
    
    def add_postrace(self):
        peaks = find_peak_1d(points_to_array(self.tracepoints),
                             tracedir = self.trace_axis, pn='neg')
        new_peak = float(peaks)
        peakheight = self.tplot.points.interp(new_peak)
        plot = SmoothLinePlot(color=[0,1,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
        self.ids.the_graph.add_plot(plot)
        newspin = ApertureSlider(aperture_line = plot, tfscreen = self)
        newspin.slider.range = [0, len(self.tracepoints)-1]
        newspin.slider.step = 0.1
        newspin.slider.value = new_peak
        newspin.trash.bind(on_press = lambda x: self.remtrace('pos',newspin))
        self.ids.postrace.add_widget(newspin)
        self.apertures['pos'].append(newspin)
        
    def add_negtrace(self):
        peaks = find_peak_1d(points_to_array(self.tracepoints),
                             tracedir = self.trace_axis, pn='neg')
        new_peak = float(peaks)
        peakheight = self.tplot.points.interp(new_peak)
        plot = SmoothLinePlot(color=[1,0,0,1], points=[(new_peak, 0), (new_peak, peakheight)])
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
        stripe = RobustData(points_to_array(self.tracepoints))
        wid = self.fit_params['wid']
        pos = {'pos':[(stripe.interp(x.slider.value), 
                       x.slider.value, wid) for x in self.apertures['pos']],
               'neg':[(stripe.interp(x.slider.value), 
                       x.slider.value, wid) for x in self.apertures['neg']]}
        for x in self.trace_lines:
            if x in self.ids.the_graph.plots:
                self.ids.the_graph.remove_plot(x)
        if self.fit_params.get('man',False):
            popup = DefineTrace(npos=len(self.apertures['pos']), \
                nneg=len(self.apertures['neg']), imtexture = self.iregion)
            popup.bind(on_dismiss = self.manual_trace(popup.tracepoints))
            popup.open()
            return
        self.xx, self.fit_params['pmodel'], self.fit_params['nmodel'] = \
            fit_peaks(stripe, pos=pos, ptype = self.fit_params['shape'])
        self.trace_lines[0].points = zip(self.xx, self.fit_params['pmodel'](self.xx))
        self.trace_lines[1].points = zip(self.xx, self.fit_params['nmodel'](self.xx))
        self.ids.the_graph.add_plot(self.trace_lines[0])
        self.ids.the_graph.add_plot(self.trace_lines[1])
    
    def fix_distort(self):
        if not (self.fit_params.get('pmodel',False) or \
            self.fit_params.get('nmodel',False)):
            popup = AlertDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        pdistort, ndistort = draw_trace(self.extractregion, self.xx, self.fit_params['pmodel'], \
            self.fit_params['nmodel'], fixdistort = True, fitdegree = self.fit_params['deg'], bin=20)
        
        im1, im2 = [x for x in copy.deepcopy(self.theapp.extract_pairs[self.pair_index])]
        im1.load(); im2.load()
        im1.data_array = fix_distortion(im1.data_array, pdistort)
        im2.data_array = undistort_imagearray(im2.data_array, ndistort)
        im_subtract(im1, im2, outputfile=self.current_impair.fitsfile)
        tmp = self.current_impair
        self.current_impair = FitsImage(self.current_impair.fitsfile)
        self.current_impair.header['EXREGX1'] = (tmp.get_header_keyword('EXREGX1'), 'extraction region coordinate X1')
        self.current_impair.header['EXREGY1'] = (tmp.get_header_keyword('EXREGY1'), 'extraction region coordinate Y1')
        self.current_impair.header['EXREGX2'] = (tmp.get_header_keyword('EXREGX2'), 'extraction region coordinate X2')
        self.current_impair.header['EXREGY2'] = (tmp.get_header_keyword('EXREGY2'), 'extraction region coordinate Y2')
        self.current_impair.update_fits(header_only = True)
        self.set_imagepair(self.pairstrings[self.pair_index])
        self.fit_params['nmodel'] = None
        self.fit_params['pmodel'] = None
        
    
    def manual_trace(self, traces):
        pass #need to figure out how to apply these
    
    def extract_spectrum(self):
        if not (self.fit_params.get('pmodel',False) or \
            self.fit_params.get('nmodel',False)):
            popup = AlertDialog(text='Make sure you fit the trace centers first!')
            popup.open()
            return
        
        #need a calibration, too
        self.lamp = None
        if self.theapp.current_night.cals:
            self.lamp = self.theapp.current_night.cals.data_array if not self.current_flats \
                else im_divide(self.theapp.current_night.cals, self.current_flats).data_array
            self.lamp = self.lamp[self.region[1]:self.region[3]+1,self.region[0]:self.region[2]+1]
        im1, im2 = [x for x in copy.deepcopy(self.theapp.extract_pairs[self.pair_index])]
        im1.load(); im2.load()
        im1.data_array = undistort_imagearray(im1.data_array, pdistort)
        im2.data_array = undistort_imagearray(im2.data_array, ndistort)
        self.tell = make_region(im1, im2, self.region, flat=self.current_flats, telluric=True)
        self.pextract = extract(self.fit_params['pmodel'], self.extractregion, self.tell, 'pos', lamp = self.lamp)
        self.nextract = extract(self.fit_params['nmodel'], self.extractregion, self.tell, 'neg', lamp = self.lamp)
        
        #write uncalibrated spectra to fits files (will update after calibration)
        pstub = self.paths['out'] + re.sub('.fits','-ap%i',im1.fitsfile)
        ext = ('.fits','-sky.fits','-lamp.fits')
        h = im1.header
        for i, p_ap in enumerate(self.pextract):
            for j in range(p_ap.shape[1]):
                spec = p_ap[:,j]
                fits.writeto((pstub + ext[i]) % j, spec, header=h)
        
        nstub = self.paths['out'] + re.sub('.fits','-ap%i',im2.fitsfile)
        h = im2.header
        for i, n_ap in enumerate(self.nextract):
            for j in range(n_ap.shape[1]):
                spec = n_ap[:,j]
                fits.writeto((nstub + ext[i]) % j, spec, header=h)
