import kivy
kivy.require('1.8.0')

from kivy.app import App
from kivy.uix.screenmanager import FadeTransition
from kivy.properties import (ListProperty, NumericProperty, ObjectProperty,
                             StringProperty, DictProperty)
from dialogs import AlertDialog
from datatypes import ObsNight, ObsTarget
from screens import (InstrumentScreen, ObservingScreen, ExtractRegionScreen,
                     TracefitScreen, WavecalScreen, CombineScreen, 
                     TelluricScreen)

class IRReduceApp(App):
    transition = ObjectProperty(FadeTransition())
    current_title = StringProperty('')
    index = NumericProperty(-1)
    screen_names = ListProperty([])
    extract_pairs = ListProperty([])
    current_night = ObjectProperty(ObsNight())
    current_extraction = ObjectProperty(None)
    current_target = ObjectProperty(ObsTarget())
    current_paths = DictProperty({})
    current_dithers = ListProperty([])
    current_region = ListProperty([])
    current_flats = ObjectProperty(None, force_dispatch=True, allownone=True)
    rdb = ObjectProperty(None)
    
    def build(self):
        self.title = 'Spectroscopic Pipeline for Interactive Data Reduction (SPIDR)'
        self.icon = 'resources/irreduc-icon.png'
        self.screen_names = ['Instrument Profile',
            'Observing Run', 'Extraction Region',
            'Trace Fitting', 'Wavelength Calibration',
            'Combine Spectra', 'Telluric Correction']
        self.shortnames = ['instrument','observing',
            'region','trace','wavecal','combine','telluric']
        sm = self.root.ids.sm
        sm.add_widget(InstrumentScreen())
        sm.add_widget(ObservingScreen())
        sm.add_widget(ExtractRegionScreen())
        sm.add_widget(TracefitScreen())
        sm.add_widget(WavecalScreen())
        sm.add_widget(CombineScreen())
        sm.add_widget(TelluricScreen())
        sm.current = 'instrument'
        self.index = 0
        self.current_title = self.screen_names[self.index]
    
    def on_pause(self):
        return True

    def on_resume(self):
        pass
    
    def on_current_title(self, instance, value):
        self.root.ids.spnr.text = value
    
    def go_previous_screen(self):
        self.index = (self.index - 1) % len(self.screen_names)
        self.update_screen()
    
    def go_next_screen(self): 
        if (self.root.ids.sm.current == 'observing' and 
            self.root.ids.sm.screens[1].current_target.targid == ''):
            popup = AlertDialog(text='You need to select a target before proceeding!')
            popup.open()
            return
        self.index = (self.index + 1) % len(self.screen_names)
        self.update_screen()
        
    def go_screen(self, idx):
        if (idx > 1 and self.root.ids.sm.screens[1].current_target.targid == ''):
            popup = AlertDialog(text='You need to select a target before proceeding!')
            popup.open()
            return
        self.index = idx
        self.update_screen()
    
    def update_screen(self):
        self.root.ids.sm.current = self.shortnames[self.index]
        self.current_title = self.screen_names[self.index]
    
    def save_current(self):
        if self.current_extraction: self.current_target.extractions[self.current_extraction.name] = self.current_extraction
        self.current_night.targets[self.current_target.targid] = self.current_target
        self.rdb[self.current_night.date] = self.current_night

if __name__ == '__main__':
    IRReduceApp().run()