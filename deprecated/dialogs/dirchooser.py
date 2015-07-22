# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:42:11 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
import os

Builder.load_string('''
<DirChooser>:
    title: 'Choose a Directory'
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        FileChooserListView:
            id: chooser
            dirselect: True
            multiselect: False
            filter_dirs: True
            filters: [root.check_dir]
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '50dp'
            Button:
                text: 'Cancel'
                on_press: root.dismiss()
            Button:
                text: 'Select'
                on_press: root.set_directory()
''')


class DirChooser(Popup):
    chosen_directory = StringProperty('')
    
    def check_dir(self, folder, filename):
        return os.path.isdir(os.path.join(folder, filename))
    
    def set_directory(self):
        if not self.ids.chooser.selection:
            return
        self.chosen_directory = self.ids.chooser.selection[0]
        self.dismiss()