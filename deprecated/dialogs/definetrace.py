# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:30:25 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import (DictProperty, ListProperty, NumericProperty,
                             ObjectProperty)
from kivy.graphics.vertex_instructions import Point

Builder.load_string('''
<DefineTrace>:
    title: 'Manually define your trace'
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        Widget:
            id: the_widget
            size_hint: 1, 0.9
            canvas.before:
                Rectangle:
                    size: self.size
                    pos: self.pos
                    texture: root.imtexture
            canvas.after:
                Color:
                    rgba: 1, 0, 0, 1
        BoxLayout:
            orientation: 'horizontal'
            Spinner:
                size_hint_x: 0.5
                text: 'Choose an aperture'
                values: ['Positive aperture %i' % x for x in range(root.npos)] + \
                    ['Negative aperture %i' % x for x in range(root.nneg)]
                on_text: root.set_aperture(self.values.index(self.text))
            Button:
                size_hint_x: 0.25
                text: 'Cancel'
                on_press: root.dismiss()
            Button:
                size_hint_x: 0.25
                text: 'Submit'
                on_press: root.set_traces()
''')

class DefineTrace(Popup):
    imtexture = ObjectProperty(None)
    npos = NumericProperty(0)
    nneg = NumericProperty(0)
    traces = ListProperty([])
    ap_index = NumericProperty(0)
    tracepoints = DictProperty({})
    
    def on_open(self):
        self.tracepoints = {'pos':[], 'neg':[]}
        self.traces = [('pos', Point(pointsize=2)) for i in range(self.npos)] + \
            [('neg', Point(pointsize=2)) for i in range(self.nneg)]
    
    def set_aperture(self,new_ind):
        self.ids.the_widget.canvas.remove(self.traces[self.ap_index])
        self.ap_index = new_ind
        self.ids.the_widget.canvas.add(self.traces[self.ap_index])
    
    def set_traces(self):
        for point in self.traces:
            thepoints = [self.ids.the_widget.to_widget(x) \
                for x in point[1].points]
            self.tracepoints[point[0]].append(thepoints)
        
    def on_touch_down(self, touch):
        if self.ids.the_widget.collide_point(touch):
            self.traces[self.ap_index][1].add_point(touch)
            return True