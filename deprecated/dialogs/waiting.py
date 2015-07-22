# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:27:48 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import StringProperty

Builder.load_string('''
<WaitingDialog>:
    title: 'Please Wait'
    auto_dismiss: False
    size_hint: 0.8, 0.8
    BoxLayout:
        orientation: 'vertical'
        Widget:
        AnchorLayout:
            size_hint: 1, 1
            AsyncImage:
                id: loadimage
                size_hint: None, None
                size: '128dp', '128dp'
                source: 'resources/loading_icon.gif'
        Label:
            size_hint_y: 0.2
            text: root.text
            text_size: self.size
            halign: 'center'
        Widget:
''')

class WaitingDialog(Popup):
    text = StringProperty('')