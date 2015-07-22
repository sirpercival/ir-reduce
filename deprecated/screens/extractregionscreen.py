# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:22:34 2015

@author: gray
"""

from irscreen import IRScreen
from kivy.lang import Builder
from kivy.properties import (AliasProperty, DictProperty, ListProperty, 
                             NumericProperty, ObjectProperty)
from custom_widgets.imagepane import default_image
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from datatypes import FitsImage
from image_arithmetic import im_subtract
import os, re, copy
from astropy.io import fits
from dialogs import AlertDialog

Builder.load_string('''
<ExtractRegionScreen>:
    name: 'region'
    imcanvas: ipane.ids.the_image
    theapp: app
    paths: app.current_paths
    extract_pairs: app.extract_pairs
    BoxLayout:
        orientation: 'vertical'
        size_hint_x: 0.4
        Spinner:
            text: 'Choose an image pair:'
            values: root.pairstrings
            on_text: root.set_imagepair(self.text)
            size_hint: 1, 0.1
        BorderBox:
            size_hint: 1, 0.4
            BoxLayout:
                orientation: 'vertical'
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1, 0.25
                    Label:
                        text: 'X1:'
                        size_hint: 0.4, 1
                    TextInput:
                        id: x1t
                        text: str(root.bx1)
                        multiline: False
                        on_focus: if not self.focus: root.set_coord('x1', self.text)
                Slider:
                    id: x1s
                    min: 0
                    max: root.imwid
                    step: 1
                    value: root.bx1
                    size_hint: 1, 0.25
                    on_value: root.set_coord('x1', self.value)
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1, 0.25
                    Label:
                        text: 'Y1:'
                        size_hint: 0.4, 1
                    TextInput:
                        id: y1t
                        text: str(root.by1)
                        multiline: False
                        on_focus: if not self.focus: root.set_coord('y1', self.text)
                Slider:
                    id: y1s
                    min: 0
                    max: root.imwid
                    step: 1
                    value: root.by1
                    size_hint: 1, 0.25
                    on_value: root.set_coord('y1', self.value)
        BorderBox:
            size_hint: 1, 0.4
            BoxLayout:
                orientation: 'vertical'
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1, 0.25
                    Label:
                        text: 'X2:'
                        size_hint: 0.4, 1
                    TextInput:
                        id: x2t
                        text: str(root.bx2)
                        multiline: False
                        on_focus: if not self.focus: root.set_coord('x2', self.text)
                Slider:
                    id: x2s
                    min: 0
                    max: root.imwid
                    step: 1
                    value: root.bx2
                    size_hint: 1, 0.25
                    on_value: root.set_coord('x2', self.value)
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1, 0.25
                    Label:
                        text: 'Y2:'
                        size_hint: 0.4, 1
                    TextInput:
                        id: y2t
                        text: str(root.by2)
                        multiline: False
                        on_focus: if not self.focus: root.set_coord('y2', self.text)
                Slider:
                    id: y2s
                    min: 0
                    max: root.imwid
                    step: 1
                    value: root.by2
                    size_hint: 1, 0.25
                    on_value: root.set_coord('y2', self.value)
        Button:
            text: 'Select Extraction Region'
            size_hint: 1, 0.1
            on_press: root.save_region()
    ImagePane:
        id: ipane
        size_hint_x: 0.6
''')

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
            Color(30./255., 227./255., 224./255.)
            self.regionline = Line(points=self.lcoords, close=True, \
                dash_length = 2, dash_offset = 1)

    def on_enter(self):
        flat = os.path.join(self.paths['cal'],'Flat.fits')
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
        self.pairstrings = ['{0} - {1}'.format(*[os.path.basename(x.fitsfile) for x in y]) for y in self.extract_pairs]
    
    def on_pre_leave(self):
        self.theapp.current_impair = self.current_impair
        self.theapp.current_flats = self.current_flats
    
    def set_imagepair(self, val):
        if not self.theapp.current_target:
            popup = AlertDialog(text='You need to select a target (on the Observing Screen) before proceeding!')
            popup.open()
            return
        pair_index = self.pairstrings.index(val)
        fitsfile = self.paths['out']+re.sub(' ','',re.sub('.fits','',val))+'.fits'
        if not os.path.isfile(fitsfile):
            im1, im2 = [x for x in copy.deepcopy(self.extract_pairs[pair_index])]
            im1.load(); im2.load()
            im_subtract(im1, im2, outputfile=os.path.join(self.paths['out'],fitsfile))
        self.current_impair = FitsImage(os.path.join(self.paths['out'],fitsfile), load=True)
        self.ids.ipane.load_data(self.current_impair)
        self.imwid, self.imht = self.current_impair.dimensions
        if self.current_impair.get_header_keyword('EXREGX1'):
            for x in ['x1', 'y1', 'x2', 'y2']:
                tmp = self.current_impair.get_header_keyword('EXREG'+x.upper())
                if tmp[0] is not None:
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