# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:53:27 2015

@author: gray
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import AliasProperty, ListProperty, NumericProperty, ObjectProperty
from kivy.lang import Builder
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.graphics.texture import Texture
from datatypes import ScalableImage
import numpy as np

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

<ImagePane>:
    orientation: 'vertical'
    imcanvas: the_image.canvas
    widsize: the_image.size
    StencilView:
        id: sv
        size_hint: 1., 0.8
        BoxLayout:
            size: sv.size
            pos: sv.pos
            padding: 20
            orientation: 'horizontal'
            FloatLayout:
                id: fl
                ScatterLayout:
                    pos: fl.pos
                    id: the_scatter
                    do_rotation: False
                    do_translation: self.scale != 1
                    Widget:
                        id: the_image
                        canvas.before:
                            Rectangle:
                                size: root.rectsize
                                pos: [self.center[i] - root.rectsize[i]/2 for i in range(2)]
                                texture: root.iregion
    GridLayout:
        size_hint: 1., 0.2
        spacing: 2
        cols: 2
        Label:
            text: 'Stretch mode:'
        Label:
            text: 'Stretch factor:'
        Spinner:
            id: smode
            text: 'linear'
            values: 'linear', 'logarithmic', 'gamma', 'arcsinh', 'square root', 'histogram equalization'
            on_text: root.update_slider()
        Slider:
            id: sfactor
            value_normalized: 0.5
            disabled: True
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: 'Reset Zoom'
                on_press: root.reset_zoom()
            Button:
                text: 'Submit'
                on_press: root.submit_changes()
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Min'
            TextInput:
                id: ti_min
                text: '%f3' % root.tmin
                multiline: False
            Label:
                text: 'Max'
            TextInput:
                id: ti_max
                text: '%f3' % root.tmax
                multiline: False

<MyGraph>:
''')



tmp = np.array([ x / 8 + 1 for x in range(1024)], dtype=np.float64)
data = tmp + tmp.reshape(-1,1)
default_image = ScalableImage()
default_image.load(data, scalemode='gamma', factor=0.3)
        
class ImagePane(BoxLayout):
    data = ObjectProperty(default_image)
    itexture = ObjectProperty(Texture.create(size = (2048, 2048)))
    iregion = ObjectProperty(None)
    tmin = NumericProperty(0.)
    tmax = NumericProperty(0.)
    imcanvas = ObjectProperty(None)
    widsize = ListProperty([])
    
    def get_rectsize(self):
        if not self.iregion:
            return self.ids.the_image.size
        regx, regy = self.iregion.size
        widx, widy = self.ids.the_image.size
        scale = min([float(widx) / float(regx), float(widy) / float(regy)])
        return int(regx * scale), int(regy * scale)
    
    rectsize = AliasProperty(get_rectsize, None, bind=('iregion', 'widsize'))
    
    def load_data(self, dataobj):
        self.data = dataobj
        self.tmin = float(self.data.threshold[0])
        self.tmax = float(self.data.threshold[1])
        self.iregion = self.itexture.get_region(0, 0, self.data.dimensions[0], \
            self.data.dimensions[1])
        self.update_display()
    
    def update_display(self):
        idata = ''.join(map(chr,self.data.scaled))
        self.itexture.blit_buffer(idata, colorfmt='luminance', bufferfmt='ubyte', \
            size = self.data.dimensions)
    
    def update_slider(self):
        mode = self.ids.smode.text
        factor = self.ids.sfactor
        if mode in ['linear', 'histogram equalization', 'square root']:
            factor.disabled = True
        elif mode == 'gamma':
            factor.disabled = False
            val = factor.value_normalized
            factor.range = (-1, 1)
            factor.step = 0.05
            factor.value_normalized = val
        elif mode == 'logarithmic':
            factor.disabled = False
            val = factor.value_normalized
            factor.range = (1, 10)
            factor.step = 0.25
            factor.value_normalized = val
        elif mode == 'arcsinh':
            factor.disabled = False
            val = factor.value_normalized
            factor.range = (1, 6)
            factor.step = 0.1
            factor.value_normalized = val
    
    def reset_zoom(self):
        scatr = self.ids.the_scatter
        scatr.pos = self.ids.fl.pos
        scatr.scale = 1
    
    def submit_changes(self):
        self.tmin = float(self.ids.ti_min.text)
        self.tmax = float(self.ids.ti_max.text)
        factor = self.ids.sfactor.value
        if self.ids.smode.text == 'gamma':
            factor = 10.**factor
        info = {'min':self.tmin, 'max':self.tmax, \
            'mode':self.ids.smode.text, 'factor':factor}
        self.data.change_parameters(info)
        self.update_display()

class ComboEdit(TextInput):

    options = ListProperty(('', ))

    def __init__(self, **kw):
        ddn = self.drop_down = DropDown()
        ddn.bind(on_select=self.on_select)
        super(ComboEdit, self).__init__(**kw)

    def on_options(self, instance, value):
        ddn = self.drop_down
        ddn.clear_widgets()
        for widg in value:
            widg.bind(on_release=lambda btn: ddn.select(btn.text))
            ddn.add_widget(widg)

    def on_select(self, *args):
        self.text = args[1]
        self.focus = True
        self.focus = False

    def on_touch_up(self, touch):
        if touch.grab_current == self:
            self.drop_down.open(self)
        return super(ComboEdit, self).on_touch_up(touch)

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