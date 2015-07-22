# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:34:10 2015

@author: gray
"""

from irscreen import IRScreen
from kivy.lang import Builder
from kivy.properties import (BooleanProperty, DictProperty, ListProperty, 
                             NumericProperty, ObjectProperty, StringProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import SmoothLinePlot
import glob
from datatypes import ExtractedSpectrum, SpectrumStack
from astropy.io import fits
from scipy.constants import golden
from colorsys import hsv_to_rgb
from random import random

def gen_colors(n):
    '''generate a list of dissimilar colors'''
    h = random()
    for x in xrange(n):
        h += golden
        h %= 1
        yield hsv_to_rgb(h, 0.99, 0.99)

Builder.load_string('''
<SpecscrollInsert>:
    orientation: 'horizontal'
    size_hint: 1, None
    height: '30dp'
    Label:
        text: root.text
    Checkbox:
        size_hint: None, None
        height: '30dp'
        width: '30dp'
        active: root.active
        on_active: root.active = self.active

<CombineScreen>:
    name: 'combine'
    id: combine
    speclist: [re.sub('.fits','',x.fitsfile) for x in app.current_target.images]
    paths: app.current_paths
    BoxLayout:
        orientation: 'vertical'
        Graph:
            id: multispec
            size_hint_y: 0.35
            xlabel: 'Wavelength'
            ylabel: 'Counts'
            x_ticks_minor: 5
            x_ticks_major: int((root.wmax-root.wmin)/5.)
            y_ticks_minor: 5
            y_ticks_major: int((root.dmax - root.dmin) / 5.)
            x_grid_label: True
            y_grid_label: True
            xlog: False
            ylog: False
            x_grid: False
            y_grid: False
            xmin: root.wmin
            xmax: root.wmax
            ymin: root.dmin
            ymax: root.dmax
            label_options: {'bold': True}
        Graph:
            id: combspec
            size_hint_y: 0.35
            xlabel: 'Wavelength'
            ylabel: 'Counts'
            x_ticks_minor: 5
            x_ticks_major: int((root.wmax-root.wmin)/5.)
            y_ticks_minor: 5
            y_ticks_major: int((root.dmax - root.dmin) / 5.)
            x_grid_label: True
            y_grid_label: True
            xlog: False
            ylog: False
            x_grid: False
            y_grid: False
            xmin: root.wmin
            xmax: root.wmax
            ymin: root.dmin
            ymax: root.dmax
            label_options: {'bold': True}
        BoxLayout:
            size_hint_y: 0.3
            orientation: 'horizontal'
            BorderBox:
                ScrollView:
                    BoxLayout:
                        id: specscroll
                        orientation: 'vertical'
            BoxLayout:
                orientation: 'vertical'
                Spinner:
                    text: 'Choose a fiducial for scaling'
                    values: root.speclist
                    on_text: root.set_scale(self.text)
                BoxLayout:
                    orientation: 'horizontal'
                    ToggleButton:
                        id: median
                        text: 'Median'
                        group: 'medmean'
                        state: 'down'
                        on_state: if state == 'down': root.comb_method = 'median'
                    ToggleButton:
                        text: 'Weighted Mean'
                        group: 'medmean'
                        state: 'normal'
                        on_state: if state == 'down': root.comb_method = 'mean'
                BoxLayout:
                    orientation: 'horizontal'
                    TextInput:
                        id: savefile
                        size_hint_x: 0.7
                        text: app.current_target.targid+'.fits'
                    Button:
                        size_hint_x: 0.3
                        text: 'Save'
                        on_press: root.combine()
''')

class SpecscrollInsert(BoxLayout):
    active = BooleanProperty(True)
    text = StringProperty('')
    spectrum = ObjectProperty(SmoothLinePlot())

class CombineScreen(IRScreen):
    speclist = ListProperty([])
    paths = DictProperty({})
    wmin = NumericProperty(0)
    wmax = NumericProperty(1024)
    dmin = NumericProperty(0)
    dmax = NumericProperty(1024)
    combined_spectrum = ObjectProperty(SmoothLinePlot(color=[1,1,1,1]))
    the_specs = ListProperty([])
    spec_inserts = ListProperty([])
    comb_method = StringProperty('median')
    scaled_spectra = ListProperty([])
    
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
        fits.writeto(out, zip(*self.combined_spectrum.points), header=h)