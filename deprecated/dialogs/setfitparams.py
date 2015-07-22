# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:54:45 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import DictProperty

Builder.load_string('''
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
                    text: '5'
                    multiline: False
                Label:
                    text: 'Degree of fit:'
                TextInput:
                    id: fit_deg
                    text: '2'
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
''')

class SetFitParams(Popup):
    fit_args = DictProperty({})
    
    def on_open(self):
        if self.fit_args:
            self.ids.fit_psf.text = self.fit_args['shape']
            self.ids.fit_wid.text = str(self.fit_args['wid'])
            self.ids.fit_deg.text = str(self.fit_args['deg'])  
            self.ids.manfit.text = 'Yes' if self.fit_args['man'] else 'No'
    
    def set_fit(self):
        self.fit_args = {'shape':self.ids.fit_psf.text, \
            'wid':float(self.ids.fit_wid.text), 'deg':int(self.ids.fit_deg.text),\
            'man':(self.ids.manfit.text == 'Yes')}
        self.dismiss()