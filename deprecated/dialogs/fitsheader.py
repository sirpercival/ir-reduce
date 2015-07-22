# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:34:11 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ListProperty, ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from astropy.io.fits.card import Undefined

Builder.load_string('''
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
            do_scroll_y: True
            size_hint_y: 0.9
            scroll_wheel_distance: '30dp'
            BoxLayout:
                id: box
                orientation: 'vertical'
                size_hint_y: None
        Button:
            text: 'Save & Close'
            on_press: root.update_header()
            size_hint_y: 0.1
''')

class FitsCard(BoxLayout):
    key = StringProperty('KEY')
    value = StringProperty('Value')
    comment = StringProperty('Comment')
    

class FitsHeaderDialog(Popup):
    fitsimage = ObjectProperty(None)
    cards = ListProperty([])
    
    def on_open(self):
        h = self.fitsimage.header
        self.ids.box.height = str(30*len(h.cards))+'dp'
        for cardkey in h:
            val = ' ' if isinstance(h[cardkey], Undefined) else str(h[cardkey])
            card = FitsCard(key = cardkey, value = val, \
                comment = h.comments[cardkey])
            self.cards.append(card)
            self.ids.box.add_widget(card)
    
    def update_header(self):
        h = self.fitsimage.header
        for i, card in enumerate(self.cards):
            if card.key in h:
                if isinstance(h[card.key], Undefined):
                    val = Undefined()
                else:
                    t = type(h[card.key])
                    val = t(card.value)
            else:
                val = card.value
            self.fitsimage.header[card.key] = (val, card.comment)
        self.dismiss()
                