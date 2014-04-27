from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty, ListProperty, \
    DictProperty, NumericProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.graphics.vertex_instructions import Point
from kivy.animation import Animation
from kivy.uix.image import Image

from os import path
import gc

fhdialog = '''
<FitsCard>:
    size_hint_y: None
    height: '30dp'
    TextInput:
        multiline: False
        on_text: if len(self.text) > 8: self.text = self.text[:8]
        text: root.key
        size_hint_x: 0.2
    TextInput:
        multiline: False
        text: root.value
        size_hint_x: 0.3
    TextInput:
        multiline: False
        text: root.comment
        size_hint_x: 0.5

<FitsHeaderDialog>:
    title: self.fitsimage.fitsfile
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        ScrollView:
            do_scroll_x: False
            size_hint_y: 0.9
            BoxLayout:
                id: box
                orientation: 'vertical'
        Button:
            text: 'Save & Close'
            on_press: root.update_header()
            size_hint_y: 0.1
'''

Builder.load_string(fhdialog)

class FitsCard(BoxLayout):
    key = StringProperty('KEY')
    value = StringProperty('Value')
    comment = StringProperty('Comment')
    

class FitsHeaderDialog(Popup):
    fitsimage = ObjectProperty(None)
    cards = ListProperty([])
    
    def on_open(self):
        h = self.fitsimage.header
        for cardkey in h:
            card = FitsCard(key = cardkey, value = h[cardkey], \
                comment = h.comments[cardkey])
            self.cards.append(card)
            self.ids.box.add_widget(card)
    
    def update_header(self):
        for card in self.cards:
            self.fitsimage.header[card.key] = (card.value, card.comment)
        self.dismiss()
                
dcdialog = '''
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
'''

Builder.load_string(dcdialog)

class DirChooser(Popup):
    chosen_directory = StringProperty('')
    
    def check_dir(self, folder, filename):
        return path.isdir(path.join(folder,filename))
    
    def set_directory(self):
        if not self.ids.chooser.selection:
            return
        self.chosen_directory = self.ids.chooser.selection[0]
        self.dismiss()

atdialog = '''
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
'''

Builder.load_string(atdialog)

class AddTarget(Popup):
    target_args = DictProperty({})
    instrumentlist = ListProperty([])
    
    def set_target(self):
        self.target_args = {'id':self.ids.target_id.text, \
            'iid':self.ids.target_instrument.text, \
            'files':self.ids.target_files.text}
        self.dismiss()

fitdialog = '''
<SetFitparams>:
    title: 'Set fit parameters'
    auto_dismiss: False
    size_hint: 0.7, 0.4
    BoxLayout:
        orientation: 'vertical'
        ScrollView:
            GridLayout:
                cols: 2
                col_default_height: '40dp'
                Label:
                    text: 'PSF Shape:'
                Spinner:
                    id: fit_psf
                    text: 'Choose a fit shape'
                    values: ['Gaussian','Lorentzian']
                Label:
                    text: 'Expected FWHM (in px):'
                TextInput:
                    id: fit_wid
                    text: 5
                    multiline: False
                Label:
                    text: 'Degree of fit:'
                TextInput:
                    id: fit_deg
                    text: 2
                    multiline: False
                Label:
                    text: 'Manually define trace?'
                Spinner:
                    id: manfit
                    text: 'No'
                    values: ['No','Yes']
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '50dp'
            Button:
                text: 'Cancel'
                on_press: root.dismiss()
            Button:
                text: 'Add'
                on_press: root.set_fit()
'''

Builder.load_string(fitdialog)

class SetFitParams(Popup):
    fit_args = DictProperty({})
    
    def set_fit(self):
        self.fit_args = {'shape':self.ids.fit_psf.text, \
            'wid':self.ids.fit_wid.text, 'deg':self.ids.fit_deg.text,\
            'man':(self.ids.manfit.text == 'Yes')}
        self.dismiss()
    
wdkv = '''
<WarningDialog>:
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
'''

Builder.load_string(wdkv)

class WarningDialog(Popup):
    text = StringProperty('')
    
deftrace = '''
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
'''

Builder.load_string(deftrace)

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
        self.ids.the_widget.canvas.remove(self.traces[ap_index])
        self.ap_index = new_ind
        self.ids.the_widget.canvas.add(self.traces[ap_index])
    
    def set_traces(self):
        for point in self.traces:
            thepoints = [self.ids.the_widget.to_widget(x) \
                for x in point[1].points]
            self.tracepoints[point[0]].append(thepoints)
        
    def on_touch_down(self, touch):
        if self.ids.the_widget.collide_point(touch):
            self.traces[ap_index][1].add_point(touch)
            return True
            
waitingkv = '''
<WaitingDialog>:
    title: 'Please Wait'
    auto_dismiss: False
    size_hint: 0.8, 0.8
    BoxLayout:
        orientation: 'vertical'
        Widget:
        AnchorLayout:
            size_hint: 1, 1
            Image:
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
'''

Builder.load_string(waitingkv)

class WaitingDialog(Popup):
    text = StringProperty('')
