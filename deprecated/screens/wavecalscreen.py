# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:10:08 2015

@author: gray
"""

from irscreen import IRScreen
from kivy.lang import Builder
from kivy.properties import (DictProperty, ListProperty, NumericProperty,
                             ObjectProperty, StringProperty)
from kivy.uix.button import Button
from ..persistence import linelistdb
from datatypes import ExtractedSpectrum
from dialogs import AlertDialog
from kivy.garden.graph import SmoothLinePlot

Builder.load_string('''
<WavecalScreen>:
    name: 'wavecal'
    id: wvscreen
    paths: app.current_paths
    speclist: [re.sub('.fits','',x.fitsfile) for x in app.current_target.images]
    BoxLayout:
        orientation: 'vertical'
        Graph:
            id: specdisplay
            size_hint_y: 0.6
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
        GridLayout:
            cols: 2
            size_hint_y: 0.4
            BoxLayout:
                orientation: 'vertical'
                Spinner:
                    id: specspin
                    text: 'Choose a spectrum'
                    values: root.speclist
                    on_text: root.set_spectrum(self.text)
                Label:
                    text: 'Expected wavelength range:'
                BoxLayout:
                    orientation: 'horizontal'
                    Label:
                        text: 'Min:'
                    TextInput:
                        id: rmin
                        text: str(root.wmin)
                        on_focus: if not self.focus: root.set_wmin(self.text)
                    Label:
                        text: 'Max:'
                    TextInput:
                        id: rmin
                        text: str(root.wmax)
                        on_focus: if not self.focus: root.set_wmax(self.text)
                BoxLayout:
                    orientation: 'horizontal'
                    Button:
                        text: 'Calibrate'
                        on_press: root.wavecal()
                    Button:
                        text: 'Save Spectrum'
                        on_press: root.save_spectrum()
            BoxLayout:
                orientation: 'vertical'
                ComboEdit:
                    id: linelist
                    multiline: False
                    text: 'Reference line list'
                    options: root.linelist_buttons
                    on_focus: if not self.focus: root.set_linelist(self.text)
                BorderBox:
                    size_hint_y: 0.5
                    BoxLayout:
                        orientation: 'vertical'
                        ToggleButton:
                            id: lampcal
                            text: 'Use lamp calibrations'
                            group: 'caltype'
                            state: 'down'
                        ToggleButton:
                            text: 'Use sky lines'
                            group: 'caltype'
                            state: 'normal'
                BoxLayout:
                    orientation: 'horizontal'
                    Label:
                        text: '# of iterations:'
                    TextInput:
                        id: numiter
                        text: '2'
''')

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
        self.current_spectrum.plot = SmoothLinePlot(color=[.9,1,1,1])
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