from kivy.uix.popup import Popup
from kivy.properties import (DictProperty, ListProperty, NumericProperty,
                             ObjectProperty)
from kivy.lang import Builder
from kivy.uix.togglebutton import ToggleButton
from kivy.garden.graph import MeshLinePlot
from imarith import minmax
from robuststats import polyfit
from itertools import izip

Builder.load_string('''
<TraceCheck>:
    title: 'Fit the trace'
    auto_dismiss: False
    BoxLayout:
        orientation: 'horizontal'
        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.5
            BoxLayout:
                orientation: 'horizontal
                Spinner:
                    text: 'Choose an aperture'
                    values: ['Positive aperture %i' % x for x in range(root.npos)] + \
                        ['Negative aperture %i' % x for x in range(root.nneg)]
                    on_text: root.set_aperture(self.values.index(self.text))
                    size_hint_y: 0.1
                Label:
                    text: 'Degree:'
                    size_hint_x: 0.8
                TextInput:
                    text: root.fitdegree
                    on_focus: if not self.focus: root.fitdegree = self.text
                    size_hint_x: 0.4
            Scrollview:
                StackLayout:
                    orientation: 'tb-lr'
                    id: pointlist
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: 0.1
                Button:
                    text: 'Fit Trace'
                    on_press: root.fit_trace
                Button:
                    text: 'Accept Fit'
                    on_press: root.accept_fit
        Graph:
            id: fitdisplay
            size_hint: 1, 1
            xlabel: 'X coord'
            ylabel: 'Y coord'
            x_ticks_minor: 5
            x_ticks_major: int((root.xmax - root.xmin)/5.)
            y_ticks_minor: 5
            y_ticks_major: int((root.ymax - root.ymin) / 5.)
            x_grid_label: True
            y_grid_label: True
            xlog: False
            ylog: False
            x_grid: False
            y_grid: False
            xmin: root.wmin
            xmax: root.wmax
            ymin: root.dmin
            ymax: root.dmax
            label_options: {'bold': True}
''')
    
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
        self.xmin, self.xmax = minmax(x1+x2)
        self.ymin, self.ymax = minmax(y1+y2)
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
        x, y = zip(*[x for i,x in self.tracepoints[self.ap_index] \
            if self.traceselectors[self.ap_index][i]])
        newfit = polyfit(x, y, self.fitdegree)
        self.thefit.points = izip(x, newfit(x))
    
    def accept_fit(self):
        pass