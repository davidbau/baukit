from .labwidget import Img, Property
import inspect

class PlotWidget(Img):
    """
    A widget to create interactive matplotlib plots by defining a simple function.
    Example of usage:

    ```
		import numpy
		def simple_redraw_rule(fig, amp=1.0, freq=1.0):
			[ax] = fig.axes
			ax.clear()
			x = numpy.linspace(0, 5, 100)
			ax.plot(x, amp * numpy.sin(freq * x))
							   
		plot = PlotWidget(simple_redraw_rule)
		display(plot)
    ```

    The keyword arguments in the provided function will become properties
    of the plot widget; updating those properties will redraw the plot in-place.
    For example, in the above, assigning `plot.freq = 3` will redraw the
    plot with freq set to 3.
    """
    def __init__(self, redraw_rule, rc=None, **kwargs):
        import matplotlib, matplotlib.pyplot
        super().__init__()
        init_args = dict(kwargs)
        self.render_args = dict(format='svg') # Looks better; use as default.
        if rc is None:
            rc = {}
        self.rc = rc
        
        all_names = []
        has_fig_argument = False
        for i, (name, p) in enumerate(inspect.signature(redraw_rule).parameters.items()):
            if i == 0 and p.default == inspect._empty:
                # assert p.default == inspect._empty, 'First arg of redraw rule should be the figure'
                has_fig_argument = True
            else:
                if name in init_args:
                    default = init_args.pop(name)
                else:
                    assert p.default != inspect._empty, 'Arguments must have default values'
                    default = p.default
                setattr(self, name, Property(default))
                all_names.append(name)

        for name in ['format', 'metadata', 'bbox_inches', 'pad_inches',
                'facecolor', 'edgecolor', 'backend']:
            if name in init_args:
                self.render_args[name] = init_args.pop(name)

        for default_arg, default_value in [('figsize', (5, 3.5))]:
            if default_arg not in init_args:
                init_args[default_arg] = default_value

        with matplotlib.pyplot.rc_context(rc=self.rc):
            old_backend = matplotlib.pyplot.get_backend()
            matplotlib.pyplot.switch_backend('agg')
            if 'mosaic' in init_args:
               self.fig, _ = matplotlib.pyplot.subplot_mosaic(**init_args)
            else:
               self.fig, _ = matplotlib.pyplot.subplots(**init_args)
        matplotlib.pyplot.switch_backend(old_backend)

        def invoke_redraw():
            with matplotlib.pyplot.rc_context(rc=self.rc):
                args = [self.fig] if has_fig_argument else []
                for name in all_names:
                    args.append(getattr(self, name))
                if not has_fig_argument:
                    matplotlib.pyplot.figure(self.fig)
                redraw_rule(*args)
                self.render(self.fig, **self.render_args)
                if not has_fig_argument:
                    matplotlib.pyplot.close(self.fig)
        self.on(' '.join(all_names), invoke_redraw)
        self._redraw = invoke_redraw
        self.redraw()

    def redraw(self):
        self._redraw()

    def event_location(self, event):
        '''
        Transform a click event from pixel coordinates to plot data coordinates.
        '''
        # Image natural size and image-relative pixel location.
        w = event.value['width']
        h = event.value['height']
        px = event.value['x']
        py = event.value['y']
        # To convert from image to display coords, we need the bbox.
        # https://stackoverflow.com/questions/28692981
        bbox_inches = self.render_args.get('bbox_inches', self.rc.get('savefig.bbox', None))
        if bbox_inches is None:
            bbox = self.fig.bbox
        else:
            padding = self.render_args.get('pad_inches', self.rc.get('savefig.pad_inches', 0.1))
            if bbox_inches == 'tight':
                bbox_inches = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            bbox = bbox_inches.padded(padding)
            bbox.set_points(bbox.get_points() * self.fig.dpi)
        # Matplotlib display-coordinate location
        dx = (bbox.x1 - bbox.x0) * px / w + bbox.x0
        dy = (bbox.y1 - bbox.y0) * (h - py) / h + bbox.y0
        # Identify the first axis that contains the point
        for inside in self.fig.axes:
            if inside.get_window_extent().contains(dx, dy):
                break
        else:
            inside = None
        ax = self.fig.axes[0] if (len(self.fig.axes) and inside is None) else inside
        # Axis-data-relative coordinate location.
        # https://stackoverflow.com/questions/59794014
        if ax is not None:
            x, y = ax.transData.inverted().transform((dx, dy))
        else:
            x, y = None, None

        class PlotLocation():
            def __init__(self, x, y, axis, inside):
                self.x = x
                self.y = y
                self.axis = axis
                self.inside = inside
        return PlotLocation(x, y, ax, inside is not None)
