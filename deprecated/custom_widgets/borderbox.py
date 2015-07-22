# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:32:34 2015

@author: gray
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ObjectProperty
from kivy.lang import Builder

Builder.load_string('''
<BorderBox>:
    padding: self.borderweight
    the_container: anchorchild
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            size: self.size
            pos: self.pos
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            size: (x - self.borderweight for x in self.size)
            pos: self.x + self.borderweight, self.y + self.borderweight
    AnchorLayout:
        id: anchorchild
''')

class BorderBox(BoxLayout):
    borderweight = NumericProperty(2)
    the_container = ObjectProperty(None)
    
    def add_widget(self, widget):
        if self.the_container is None:
            return super(BorderBox, self).add_widget(widget)
        return self.the_container.add_widget(widget)
    
    def remove_widget(self, widget):
        if self.the_container:
            return self.the_container.remove_widget(widget)
        return super(BorderBox, self).remove_widget(widget)