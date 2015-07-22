# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:49:47 2015

@author: gray
"""

from kivy.uix.screenmanager import Screen
from kivy.properties import BooleanProperty
from kivy.lang import Builder

Builder.load_string('''
<IRScreen>:
    ScrollView:
        do_scroll_x: False
        do_scroll_y: False if root.fullscreen else (content.height > root.height - dp(16))
        AnchorLayout:
            size_hint_y: None
            height: root.height if root.fullscreen else max(root.height, content.height)
            GridLayout:
                id: content
                cols: 2
                spacing: '8dp'
                padding: '8dp'
                size_hint: (1, 1) if root.fullscreen else (.9, None)
                height: self.height if root.fullscreen else root.height
''')

class IRScreen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(IRScreen, self).add_widget(*args)