# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:00:15 2015

@author: gray
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.garden.graph import MeshLinePlot
from kivy.properties import (DictProperty, ListProperty, NumericProperty, 
                             ObjectProperty, StringProperty)
from kivy.graphics.vertex_instructions import Point
import os
from astropy.io import fits
from astropy.modeling.models import Polynomial1D
from itertools import izip
from datatypes import fit_to_model, replace_nans
import numpy as np
from custom_widgets import BorderBox

Builder.load_file('dialogs.kv')

class AddTarget(Popup):
    target_args = DictProperty({})
    instrumentlist = ListProperty([])
    
    def set_target(self):
        self.target_args = {'targid':self.ids.target_id.text, \
            'instrument_id':self.ids.target_instrument.text, \
            'filestring':self.ids.target_files.text}
        self.dismiss()

class AlertDialog(Popup):
    text = StringProperty('')

active_color = [0,1,1,1]
inactive_color = [0,0,1,1]

class AssignLines(Popup):
    lines = ListProperty([])
    active_line = NumericProperty(-1)
    assignment = ListProperty([])
    spectrum = ObjectProperty(None)
    synth = ListProperty([])
    exp_lo = NumericProperty(0.)
    exp_hi = NumericProperty(1.)
    
    def __init__(self, *args, **kw):
        super(AssignLines, self).__init__(*args, **kw)
        if self.spectrum is not None:
            spec = MeshLinePlot(color=[1,1,0,1], 
                                points=zip(self.spectrum.wav, self.spectrum.spec))
            self.ids.graph.add_plot(spec)
        if self.synth:
            ok = np.logical_and(np.greater_equal(self.synth[0], self.exp_lo),
                                np.less_equal(self.synth[0], self.exp_hi))
            self.synth = [x[ok] for x in self.synth]
            normwav = self.synth[1] / replace_nans(self.synth[1]).max()
            syn = MeshLinePlot(color=[1,0,1,1],
                               points=zip(self.synth[0], normwav))
            self.ids.synth.add_plot(syn)
    
    def add_line(self):
        g, s = self.ids.graph, self.ids.synth
        x1 = (g.xmax + g.xmin)/2.
        x2 = (s.xmax + s.xmin)/2.
        spec = MeshLinePlot(color=active_color, points=[(x1, g.ymin), (x1, g.ymax)])
        synt = MeshLinePlot(color=active_color, points=[(x2, s.ymin), (x2, s.ymax)])
        g.add_plot(spec)
        s.add_plot(synt)
        self.lines.append((spec, synt))
        self.active_line = len(self.lines) - 1
    
    def remove_line(self):
        g, s = self.ids.graph, self.ids.synth
        spec, synt = self.lines[self.active_line]
        g.remove_plot(spec)
        s.remove_plot(synt)
        del self.lines[self.active_line]
        self.active_line -= 1

    def next_line(self):
        self.active_line = (self.active_line + 1) % len(self.lines)
    
    def assign(self):
        if len(self.lines) < 3:
            popup = AlertDialog(text='Please assign at least 3 lines!')
            popup.open()
            return
        self.assignment = [(g.points[0][0], s.points[0][0]) for g, s in self.lines]
        self.dismiss()
    
    def on_active_line(self, instance, value):
        for i, (g, s) in enumerate(self.lines):
            if i == value:
                g.color = active_color
                s.color = active_color
            else:
                g.color = inactive_color
                s.color = inactive_color
    
    def move_line(self, x, y):
        g, s = self.ids.graph, self.ids.synth
        line = self.lines[self.active_line]
        gx, gy = x - g.pos[0], y - g.pos[1]
        sx, sy = x - s.pos[0], y - s.pos[1]
        if g.collide_plot(gx, gy):
            dx, dy = g.to_data(gx, gy)
            line[0].points = [(dx, g.ymin), (dx, g.ymax)]
        elif s.collide_plot(sx, sy):
            dx, dy = s.to_data(sx, sy)
            line[1].points = [(dx, s.ymin), (dx, s.ymax)]
    
    def on_touch_down(self, touch):
        if self.active_line < 0:
            return super(AssignLines, self).on_touch_down(touch)
        self.move_line(*touch.pos)
        return super(AssignLines, self).on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.active_line < 0:
            return super(AssignLines, self).on_touch_move(touch)
        self.move_line(*touch.pos)
        return super(AssignLines, self).on_touch_move(touch)

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
            return super(DefineTrace, self).on_touch_down(touch)
        

class DirChooser(Popup):
    chosen_directory = StringProperty('')
    
    def check_dir(self, folder, filename):
        return os.path.isdir(os.path.join(folder, filename))
    
    def set_directory(self):
        if not self.ids.chooser.selection:
            return
        self.chosen_directory = self.ids.chooser.selection[0]
        self.dismiss()

class ExamineSpectrum(Popup):
    target = ObjectProperty(None)
    spectrum = ObjectProperty(None, force_dispatch=True, allownone=True)
    header = ObjectProperty(None)
    outfile = StringProperty('')
    
    def on_open(self):
        plot = MeshLinePlot(color=[0,1,1,1])
        plot.points = zip(np.arange(self.spectrum.size), self.spectrum)
        self.ids.graph.add_plot(plot)
    
    def save(self):
        fits.writeto(self.outfile, self.spectrum, header=self.header, clobber=True)
        if not self.outfile in self.target.spectra:
            self.target.spectra.append(self.outfile)
        self.dismiss()

class FitsCard(BoxLayout):
    key = StringProperty('KEY')
    value = StringProperty('Value')
    comment = StringProperty('Comment')

class FitsHeaderDialog(Popup):
    fitsheader = ObjectProperty(None)
    cards = ListProperty([])
    
    def on_open(self):
        h = self.fitsheader
        self.ids.box.height = str(30*len(h.cards))+'dp'
        for cardkey in h:
            val = ' ' if isinstance(h[cardkey], fits.card.Undefined) else str(h[cardkey])
            card = FitsCard(key = cardkey, value = val, \
                comment = h.comments[cardkey])
            self.cards.append(card)
            self.ids.box.add_widget(card)
    
    def update_header(self):
        h = self.fitsheader
        for i, card in enumerate(self.cards):
            if card.key in h:
                if isinstance(h[card.key], fits.card.Undefined):
                    val = fits.card.Undefined()
                else:
                    t = type(h[card.key])
                    val = t(card.value)
            else:
                val = card.value
            self.fitsheader[card.key] = (val, card.comment)
        self.dismiss()

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

class TraceCheck(Popup):
    xmin = NumericProperty(0)
    xmax = NumericProperty(1024)
    ymin = NumericProperty(0)
    ymax = NumericProperty(1024)
    npos = NumericProperty(0)
    nneg = NumericProperty(0)
    thetrace = ObjectProperty(None)
    thefit = ObjectProperty(None)
    ap_index = NumericProperty(0)
    tracepoints = DictProperty({})
    traceselectors = DictProperty({})
    polyfits = DictProperty({})
    point_buttons = ListProperty([])
    fitdegree = NumericProperty(2)
    
    def on_open(self):
        for i, t in enumerate(self.tracepoints):
            self.traceselectors[t] = [True for x in xrange(len(self.tracepoints[t]))]

    def set_aperture(self,new_ind):
        self.ids.fitdisplay.remove_plot(self.thetrace)
        self.ids.fitdisplay.remove_plot(self.thefit)
        self.ids.pointlist.clear_widgets()
        self.ap_index = new_ind
        x1, y1 = zip(*self.tracepoints[self.ap_index])
        x2, y2 = x1, self.polyfits[self.ap_index](x1)
        self.thetrace = MeshLinePlot(color=(1,0,0,1), points = [x for (i, x) in \
            self.tracepoints[self.ap_index] if self.traceselectors[self.ap_index][i]])
        self.thetrace.mode = 'points'
        self.ids.fitdisplay.add_plot(self.thetrace)
        self.thefit = MeshLinePlot(color=(0,1,0,1), points = izip(x2, y2))
        self.ids.fitdisplay.add_plot(self.thefit)
        x2, y2 = zip(*self.fitpoints[self.ap_index])
        tmp = x1 + x2
        self.xmin, self.xmax = tmp.min(), tmp.max()
        tmp = y1 + y2
        self.ymin, self.ymax = tmp.min(), tmp.max()
        self.point_buttons = []
        for i, p in enumerate(self.tracepoints[self.ap_index]):
            tmp = ToggleButton(text='X: {0:7.2d}, Y: {1:7.2d}'.format(*p), \
                on_press=self.toggle_point(i, tmp.state))
            tmp.state = 'down' if self.traceselectors[self.ap_index][i] else 'normal'
            self.point_buttons.append(tmp)
            self.ids.pointlist.add_widget(tmp)
    
    def toggle_point(self, ind, state):
        self.traceselectors[self.ap_index][ind] = True if state == 'down' else False
        self.thetrace.points = [x for (i, x) in self.tracepoints[self.ap_index] \
            if self.traceselectors[self.ap_index][i]]
    
    def fit_trace(self):
        x, y = zip(*[x for i,x in self.tracepoints[self.ap_index] if self.traceselectors[self.ap_index][i]])
        #newfit = RobustData(y, x=x).fit_to_model(Polynomial1D(degree=self.fitdegree))
        newfit = fit_to_model(Polynomial1D(degree=self.fitdegree), x, y)        
        self.thefit.points = izip(x, newfit(x))
    
    def accept_fit(self):
        pass

class WaitingDialog(Popup):
    text = StringProperty('')