# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:28:38 2015

@author: gray
"""

from irscreen import IRScreen
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (DictProperty, ListProperty, ObjectProperty, 
                             StringProperty)
from datatypes import (ObsRun, ObsNight, ObsTarget, FitsImage, 
                       parse_filestring)
from dialogs import (WaitingDialog, AlertDialog, DirChooser, 
                     AddTarget, FitsHeaderDialog)
from persistence import instrumentdb, obsrundb, AdHocDB
import re, os
from threading import Thread

Builder.load_string('''
<ObservingScreen>:
    name: 'observing'
    obsrun_list: self.obsids.keys()
    theapp: app
    BoxLayout:
        orientation: 'vertical'
        ComboEdit:
            id: obsrun
            multiline: False
            text: 'Observing Run ID'
            size_hint: 0.8, 0.125
            options: root.obsrun_buttons
            on_focus: if not self.focus: root.set_obsrun()
        ComboEdit:
            id: obsnight
            multiline: False
            text: 'Observation Date'
            size_hint: 0.8, 0.125
            options: root.obsnight_buttons
            on_focus: if not self.focus: root.set_obsnight()
        BoxLayout:
            size_hint: 1, 0.125
            TextInput:
                id: rawpath
                text: 'Path to Raw Images'
                size_hint_x: 0.6
                multiline: False
                on_focus: if not self.focus: root.setpath('raw',self.text)
            Button:
                text: 'Choose'
                size_hint_x: 0.4
                on_press: root.pick_rawpath()
        BoxLayout:
            size_hint: 1, 0.125
            TextInput:
                id: outpath
                text: 'Path to Output'
                size_hint_x: 0.6
                multiline: False
                on_focus: if not self.focus: root.setpath('out',self.text)
            Button:
                text: 'Choose'
                size_hint_x: 0.4
                on_press: root.pick_outpath()
        BoxLayout:
            size_hint: 1, 0.125
            TextInput:
                id: calpath
                text: 'Path to Calibration Images'
                size_hint_x: 0.6
                multiline: False
                on_focus: if not self.focus: root.setpath('cal',self.text)
            Button:
                text: 'Choose'
                size_hint_x: 0.4
                on_press: root.pick_calpath()
        Label:
            size_hint: 0.8, 0.125
            text: 'File format:'
            halign: 'left'
        TextInput:
            id: fformat
            text: 'obs.####'
            size_hint: 1, 0.125
            multiline: False
            on_focus: if not self.focus: root.check_filestub(self.text)
        Spinner:
            id: caltypes
            size_hint: 0.8, 0.125
            text: 'Calibration Images'
            values: ['Flats (lamps ON)', 'Flats (lamps OFF)', 'Arc Lamps']
            on_text: root.set_caltype(self.text)
        BoxLayout:
            orientation: 'horizontal'
            size_hint: 1, 0.125
            TextInput:
                id: calfiles
                text: '1-10'
                multiline: False
                size_hint_x: 0.5
                on_focus: if not self.focus: root.set_calfile(self.text)
            Label:
                id: calout
                text: 'File not yet created'
        BoxLayout:
            size_hint: 1, 0.125
            Button:
                text: 'Make Cals'
                on_press: root.make_cals()
            Button:
                text: 'Save'
                on_press: root.save_night()
        Widget:
    BoxLayout:
        orientation: 'vertical'
        Spinner:
            id: targs
            size_hint: 1, 0.1
            text: 'Targets'
            values: root.target_list
            on_text: root.set_target()
        Button:
            size_hint: 1, 0.1
            text: 'Add target'
            on_press: root.add_target()
        BorderBox:
            size_hint: 1., 0.4
            ScrollView:
                do_scroll_x: False
                effect_cls: Factory.DampedScrollEffect
                BoxLayout:
                    orientation: 'vertical'
                    id: obsfiles
        TextInput:
            id: tnotes
            multiline: True
            size_hint: 1, 0.4
            text: root.current_target.notes
        Button:
            size_hint: 1, 0.1
            text: 'Save Target'
            on_press: root.save_target()

<ObsfileInsert>:
    orientation: 'horizontal'
    size_hint: 1, 0.3
    Label:
        text: path.basename(root.obsfile.fitsfile)
    Spinner:
        id: spnr
        text: root.dithertype
        values: ['A', 'B', 'X']
        on_text: root.dithertype = self.text
        size_hint_x: 0.1
    Button:
        text: 'Header'
        on_press: root.launch_header()
        size_hint_x: 0.4
''')

class ObsfileInsert(BoxLayout):
    obsfile = ObjectProperty(None)
    dithertype = StringProperty('')

    def launch_header(self):
        self.header_viewer = FitsHeaderDialog(fitsimage = self.obsfile)
        self.header_viewer.bind(on_dismiss = self.update_header())
        self.header_viewer.open()
    
    def update_header(self):
        self.obsfile.header = self.header_viewer.fitsimage.header

class ObservingScreen(IRScreen):
    obsids = DictProperty({})
    obsrun_list = ListProperty([])
    obsrun_buttons = ListProperty([])
    current_obsrun = ObjectProperty(ObsRun())
    obsnight_list = ListProperty([])
    obsnight_buttons = ListProperty([])
    current_obsnight = ObjectProperty(ObsNight())
    instrument_list = ListProperty([])
    caltype = StringProperty('')
    target_list = ListProperty([])
    current_target = ObjectProperty(ObsTarget())
    file_list = ListProperty([])
    
    def __init__(self, **kwargs):
        super(ObservingScreen, self).__init__(**kwargs)
        self.waiting = WaitingDialog(text='Please wait while the calibration images build, thank you!')
    
    def on_enter(self):
        self.instrument_list = instrumentdb.keys()
        self.obsids = {x:obsrundb[x] for x in obsrundb}
        self.obsrun_list = obsrundb.keys()
        self.obsrun_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsrun_list]
    
    def on_pre_leave(self):
        pairs = self.current_target.ditherpairs
        self.theapp.extract_pairs = [[self.current_target.images[x] for x in y] for y in pairs]
        self.theapp.current_target = self.current_target
        self.theapp.current_paths = {'cal':self.current_obsnight.calpath, 
            'raw':self.current_obsnight.rawpath, 'out':self.current_obsnight.outpath}
        self.theapp.current_night = self.current_obsnight
    
    def set_obsrun(self):
        run_id = self.ids.obsrun.text
        if run_id not in self.obsids:
            self.rdb = AdHocDB()
            self.obsids[run_id] = self.rdb.fname
            obsrundb[run_id] = {run_id:self.rdb.fname}
        else:
            self.rdb = AdHocDB(self.obsids[run_id][run_id])
        self.current_obsrun = ObsRun(runid=run_id)
        tmp = {str(self.rdb[r][r].date):self.rdb[r][r] for r in self.rdb}
        self.current_obsrun = self.current_obsrun._replace(nights=tmp)
        self.obsnight_list = self.current_obsrun.nights.keys()
        self.obsnight_buttons = [Button(text=x, size_hint_y = None, height = 30) \
            for x in self.obsnight_list]
    
    def set_obsnight(self):
        night_id = self.ids.obsnight.text
        if night_id == '' or self.current_obsrun.runid == '' \
                        or night_id == 'Observation Date':
            return
        if night_id not in self.obsnight_list:
            self.obsnight_list.append(night_id)
            self.obsnight_buttons.append(Button(text = night_id, \
                size_hint_y = None, height = 30))
            self.current_obsnight = ObsNight(date = night_id)
            self.current_obsrun.add_to(self.current_obsnight)
        else:
            self.current_obsnight = self.current_obsrun.get_from(night_id)
            self.ids.rawpath.text = self.current_obsnight.rawpath
            self.ids.outpath.text = self.current_obsnight.outpath
            self.ids.calpath.text = self.current_obsnight.calpath
            self.ids.fformat.text = self.current_obsnight.filestub
            self.set_filelist()
        for night in self.obsnight_list:
            self.rdb[night] = {night:self.current_obsrun.get_from(night)}
        self.target_list = self.current_obsnight.targets.keys()
    
    def pick_rawpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('raw',popup.chosen_directory))
        popup.open()
    
    def setpath(self, which, dir):
        if which == 'raw':
            self.current_obsnight = self.current_obsnight._replace(rawpath=dir)
            self.ids.rawpath.text = dir
        elif which == 'out':
            self.current_obsnight = self.current_obsnight._replace(outpath=dir)
            self.ids.outpath.text = dir
        elif which == 'cal':
            self.current_obsnight = self.current_obsnight._replace(calpath=dir)
            self.ids.calpath.text = dir
        
    def pick_outpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('out',popup.chosen_directory))
        popup.open()
        
    def pick_calpath(self):
        popup = DirChooser()
        popup.bind(on_dismiss=lambda x: self.setpath('cal',popup.chosen_directory))
        popup.open()
    
    def check_filestub(self, stub):
        placeholder = '#'
        reg = placeholder+'+'
        if len(re.findall(reg, stub)) != 1:
            popup = AlertDialog(text = "File format is not valid; must use '#' as placeholder only")
            popup.open()
            return
        self.current_obsnight = self.current_obsnight._replace(filestub=stub) 
    
    def set_caltype(self, caltype):
        if caltype == 'Flats (lamps ON)':
            cout, flist = self.current_obsnight.flaton if self.current_obsnight.flaton else ('Not yet created', '')
        elif caltype == 'Flats (lamps OFF)':
            cout, flist = self.current_obsnight.flatoff if self.current_obsnight.flatoff else ('Not yet created', '')
        elif caltype == 'Arc Lamps':
            cout, flist = self.current_obsnight.cals if self.current_obsnight.cals else ('Not yet created', '')
        self.ids.calfiles.text = flist
        self.ids.calout.text = cout
    
    def set_calfile(self, flist):
        caltype = self.ids.caltypes.text
        if caltype == 'Flats (lamps ON)':
            flatfile = os.path.join(self.current_obsnight.calpath,self.current_obsnight.date+'-FlatON.fits')
            tmp = self.current_obsnight.flaton
            if tmp:
                tmp[1] = flist
            else:
                tmp = ['',flist]
            try:
                if FitsImage(flatfile).header['FILES'] == flist:
                    tmp[0] = flatfile
                    self.ids.calout.txt = flatfile
            except:
                pass
            self.current_obsnight = self.current_obsnight._replace(flaton=tmp)
        elif caltype == 'Flats (lamps OFF)':
            flatfile = os.path.join(self.current_obsnight.calpath,self.current_obsnight.date+'-FlatOFF.fits')
            tmp = self.current_obsnight.flatoff
            if tmp:
                tmp[1] = flist
            else:
                tmp = ['',flist]
            try:
                if FitsImage(flatfile).header['FILES'] == flist:
                    tmp[0] = flatfile
            except:
                pass
            self.current_obsnight = self.current_obsnight._replace(flatoff=tmp)
        elif caltype == 'Arc Lamps':
            flatfile = os.path.join(self.current_obsnight.calpath,self.current_obsnight.date+'-Wavecal.fits')
            tmp = self.current_obsnight.cals
            if tmp:
                tmp[1] = flist
            else:
                tmp = ['',flist]
            try:
                if FitsImage(flatfile).header['FILES'] == flist:
                    tmp[0] = flatfile
            except:
                pass
            self.current_obsnight = self.current_obsnight._replace(cals=tmp)
        
    def make_cals(self):
        if not self.current_obsnight.rawpath:
            return
        caltype = self.ids.caltypes.text
        flist = self.ids.calfiles.text
        self.waiting.open()
        if caltype == 'Flats (lamps ON)':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.flaton, flist, \
                self.current_obsnight.date+'-FlatON.fits'))
            t.start()
        elif caltype == 'Flats (lamps OFF)':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.flatoff, flist, \
                self.current_obsnight.date+'-FlatOFF.fits'))
            t.start()
        elif caltype == 'Arc Lamps':
            t = Thread(target = self.imstack_wrapper, args=(self.current_obsnight.cals, flist, \
                self.current_obsnight.date+'-Wavecal.fits'))
            t.start()
            
    def imstack_wrapper(self, target, flist, outp):
        raw = self.current_obsnight.rawpath
        cal = self.current_obsnight.calpath
        stub = self.current_obsnight.filestub
        imstack = parse_filestring(flist, os.path.join(raw, stub))
        imstack.medcombine(output = os.path.join(cal, outp))
        target[:] = [outp, flist]
        self.ids.calfiles.text = flist
        self.ids.calout.text = outp
        self.waiting.dismiss()
    
    def save_night(self):
        tmp = self.current_obsrun.nights
        tmp[self.current_obsnight.date] = self.current_obsnight
        self.current_obsrun = self.current_obsrun._replace(nights=tmp)
        for night in self.obsnight_list:
            self.rdb[night] = {night:self.current_obsrun.get_from(night)}
        self.rdb.store_sync()
        
    def set_target(self):
        target_id = self.ids.targs.text
        self.current_target = self.current_obsnight.targets[target_id]
        self.set_filelist()
    
    def add_target(self):
        popup = AddTarget(instrumentlist = self.instrument_list)
        popup.open()
        popup.bind(on_dismiss = lambda x: self.update_targets(popup.target_args) \
            if popup.target_args else None)

    def update_targets(self, targs):
        targs['images'] = parse_filestring(targs['filestring'], \
            os.path.join(self.current_obsnight.rawpath,
                         self.current_obsnight.filestub))
        targs['dither'] = targs['images'].dithers
        self.current_target = ObsTarget(**targs)
        tmp = self.current_obsnight.targets
        tmp[self.current_target.targid] = self.current_target
        self.current_obsnight = self.current_obsnight._replace(targets=tmp)
        self.target_list = self.current_obsnight.targets.keys()
        self.ids.targs.text = self.current_target.targid
        self.set_filelist()
        self.rdb[self.current_obsnight.date] = {self.current_obsnight.date:self.current_obsnight}
    
    def set_filelist(self):
        self.ids.obsfiles.clear_widgets()
        self.file_list = []
        for f, dither in zip(self.current_target.images, self.current_target.dither):
            tmp = ObsfileInsert(obsfile = f, dithertype = dither)
            self.file_list.append(tmp)
            self.ids.obsfiles.add_widget(tmp)
    
    def save_target(self):
        self.current_target = self.current_target._replace(dither=[x.dithertype for x in self.file_list],
            notes=self.ids.tnotes.text)
        #just make sure everything is propagating correctly
        self.current_obsnight.targets[self.current_target.targid] = self.current_target
        self.current_obsrun.nights[self.current_obsnight.date] = self.current_obsnight
        for night in self.obsnight_list:
            self.rdb[night] = self.current_obsrun.get_from(night)
        self.target_list = self.current_obsnight.targets.keys()
