from .labwidget import Image, Property
import matplotlib, matplotlib.pyplot
import inspect

class PlotWidget(Image):
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
        super().__init__()
        init_args = dict(kwargs)
        self.rc = {} if rc is None else rc
        
        all_names = []
        for i, (name, p) in enumerate(inspect.signature(redraw_rule).parameters.items()):
            if i == 0:
                assert p.default == inspect._empty, 'First arg of redraw rule should be the figure'
            else:
                if name in kwargs:
                    default = init_args.pop(name)
                else:
                    assert p.default != inspect._empty, 'Arguments must have default values'
                    default = p.default
                setattr(self, name, Property(default))
                all_names.append(name)

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
                args = [self.fig]
                for name in all_names:
                    args.append(getattr(self, name))
                redraw_rule(*args)
                self.render(self.fig)
        self.on(' '.join(all_names), invoke_redraw)
        invoke_redraw()
