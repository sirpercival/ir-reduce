# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:26:05 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import StringProperty

Builder.load_string('''
<AlertDialog>:
    title: 'Oops!'
    auto_dismiss: False
    size_hint: 0.5, 0.3
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: root.text
            text_size: self.size
        Button:
            size_hint_y: 0.2
            text: 'OK'
            on_press: root.dismiss()
''')

class AlertDialog(Popup):
    text = StringProperty('')