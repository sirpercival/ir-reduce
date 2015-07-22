# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:44:23 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import DictProperty, ListProperty

Builder.load_string('''
<AddTarget>:
    title: 'Add a target...'
    auto_dismiss: False
    size_hint: 0.7, 0.4
    BoxLayout:
        orientation: 'vertical'
        ScrollView:
            GridLayout:
                cols: 2
                col_default_height: '40dp'
                Label:
                    text: 'Target ID:'
                TextInput:
                    id: target_id
                    text: ''
                    multiline: False
                Label:
                    text: 'Instrument:'
                Spinner:
                    id: target_instrument
                    text: 'Choose an Instrument'
                    values: root.instrumentlist
                Label:
                    text: 'File #s:'
                TextInput:
                    id: target_files
                    text: ''
                    multiline: False
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '50dp'
            Button:
                text: 'Cancel'
                on_press: root.dismiss()
            Button:
                text: 'Add'
                on_press: root.set_target()
''')

class AddTarget(Popup):
    target_args = DictProperty({})
    instrumentlist = ListProperty([])
    
    def set_target(self):
        self.target_args = {'targid':self.ids.target_id.text, \
            'instrument_id':self.ids.target_instrument.text, \
            'filestring':self.ids.target_files.text}
        self.dismiss()