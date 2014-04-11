
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.slider import Slider
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, AliasProperty, ListProperty
from numpy import nanmin, nanmax

from fitsimage import FitsImage, ScalableImage
from numpy import array
tmp = array([ x / 8 + 1 for x in range(1024)])
data = tmp + tmp.reshape(-1,1)
default_image = ScalableImage()
default_image.load(data, scalemode='gamma', factor=0.3)

Builder.load_string('''
#:kivy 1.8.0

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
        ''')    
    
        
class ImagePane(BoxLayout):
    data = ObjectProperty(None)
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
            
        

if __name__ == '__main__':
    from kivy.base import runTouchApp
    wid = ImagePane()
    #wid.load_data(default_image)
    im = FitsImage('/Users/gray/Desktop/mdm2013/062413/06242013_tell203-a.fits')
    im.load()
    wid.load_data(im)
    runTouchApp(wid)