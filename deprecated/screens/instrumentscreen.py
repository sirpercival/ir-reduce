# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:51:41 2015

@author: gray
"""

from kivy.lang import Builder
from irscreen import IRScreen
from kivy.properties import ListProperty, ObjectProperty, StringProperty
from kivy.uix.button import Button
from datatypes import InstrumentProfile
from ..persistence import instrumentdb as idb


Builder.load_string('''
<InstrumentScreen>:
    name: 'instrument'
    BoxLayout:
        orientation: 'vertical'
        spacing: '10dp'
        ComboEdit:
            id: iprof
            multiline: False
            text: 'Instrument Profile ID'
            size_hint: 0.8, 0.15
            options: root.instrument_list
            on_focus: if not self.focus: root.set_instrument()
        BorderBox:
            size_hint: 0.8, 0.35
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: 'Trace Direction:'
                    halign: 'left'
                ToggleButton:
                    id: trace_h
                    text: 'Horizontal'
                    group: 'tracedir'
                    state: 'down'
                ToggleButton:
                    id: trace_v
                    text: 'Vertical'
                    group: 'tracedir'
                    state: 'normal'
        BorderBox:
            size_hint: 0.8, 0.25
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: 'Detector Dimensions:'
                    halign: 'left'
                BoxLayout:
                    orientation: 'horizontal'
                    Label:
                        text: 'X:'
                    TextInput:
                        id: xdim
                        multiline: False
                        text: '1024'
                    Label:
                        text: 'Y'
                    TextInput:
                        id: ydim
                        multiline: False
                        text: '1024'
        BorderBox:
            size_hint: 0.8, 0.6
            TextInput:
                id: idesc
                text: 'Instrument Profile Description'
        Widget:
            size_hint_y: 0.5
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: '~Fits Header Keywords~'
            font_size: 14
            size_hint: 0.8, 0.1
            halign: 'center'
        Label:
            text: 'Exposure Time:'
            size_hint: 0.8, 0.1
        TextInput:
            id: etime
            text: root.current_instrument.headerkeys['exp']
            multiline: False
            size_hint: 1, 0.1
        Label:
            text: 'Airmass:'
            size_hint: 0.8, 0.1
        TextInput:
            id: secz
            text: root.current_instrument.headerkeys['air']
            multiline: False
            size_hint: 1, 0.1
        Label:
            text: 'Image Type:'
            size_hint: 0.8, 0.1
        TextInput:
            id: itype
            text: root.current_instrument.headerkeys['type']
            multiline: False
            size_hint: 1, 0.1
        Widget:
        Button:
            text: 'Save Instrument Profile'
            size_hint: 0.8, 0.1
            on_press: root.save_instrument()
''')

class InstrumentScreen(IRScreen):
    saved_instrument_names = ListProperty([])
    saved_instruments = ListProperty([])
    instrument_list = ListProperty([])
    current_text = StringProperty('')
    current_instrument = ObjectProperty(InstrumentProfile())
    trace_direction = StringProperty('horizontal')
    
    def on_pre_enter(self):
        self.saved_instrument_names = sorted(idb.keys())
        self.saved_instruments = [idb[s][s] for s in self.saved_instrument_names]
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
        idb[new_instrument.instid] = {new_instrument.instid:new_instrument}
        self.on_pre_enter()

