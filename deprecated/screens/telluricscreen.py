# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:47:12 2015

@author: gray
"""

from kivy.lang import Builder
from irscreen import IRScreen

Builder.load_string('''
<TelluricScreen>:
    name: 'telluric'
    id: tellscreen
    BoxLayout:
        orientation: 'vertical'
        Image:
            size_hint: None, None
            size: self.texture_size
            source: 'resources/stopsign.png'
        Label:
            valign: 'middle'
            halign: 'center'
            text: 'This area is under construction'
''')

class TelluricScreen(IRScreen):
    pass