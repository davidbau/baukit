# baukit

Install using `pip install git+https://github.com/davidbau/baukit`.

Provides the `baukit` package, a kit of David's secret tools to help
with productive research prototyping with pytorch.

Includes:
 * Methods for tracing and editing internal activations in a network.
 * Interactive UI widgets for quick data exploration in a notebook.
 * Online algorithms for computing running stats in pytorch.
 * Fast and feature-rich data set objects for images and text.
 * Utilities for simplifying the task of running many batch jobs.

Full details can be found by reading the code.
Here is a partial overview:

## Trace library

`Trace`, `TraceDict`, `subsequence`, `replace_module`; these simplify
the work of analyzing and altering internal computations of deep
networks.  A short example of tracing a specific layer in `net`:

```
from baukit import Trace
with Trace(net, 'layer.name') as ret:
    _ = net(inp)
    representation = ret.output
```

Read the [nethook Trace source code](https://github.com/davidbau/baukit/blob/main/baukit/nethook.py) for more information.

## Widget library

`show` is a feature-rich alternative to Jupyter notebook `display`;
it allows for quickly producing HTML layouts by arranging data and
images in nested python arrays, and it knows how to directly display
PIL images, matplotlib figure objects, and interactive widgets.
HTML elements, attributes, and CSS styles can be controlled with
functions like `show.style(color='red')`.

```
from baukit import show
show([[show.style(color=c), c] for c in ['red', 'green', 'blue']])
```

There is a [notebook here](https://github.com/davidbau/baukit/blob/main/notebooks/using_show_and_widgets.ipynb) that shows off ways to use `show()`.

`show` works with a set of `Widget` subclasses such as, `Textbox`,
`Numberbox`, `Range`, `Menu`, `PlotWidget`, `PaintWidget` that provide
data-bound reactive objects for quickly making interactive
HTML visualizations that work in a Jupyter or Colab notebook.  For
example, instad of using `matplotlib` directly to just draw a picture
of a plot, you can lay out interactive widget:

```
from baukit import PlotWidget, Range, show
import numpy
def how_to_draw_my_plot(fig, amp=1.0, freq=1.0):
    [ax] = fig.axes
    ax.clear()
    x = numpy.linspace(0, 5, 100)
    ax.plot(x, amp * numpy.sin(freq * x))
						   
plot = PlotWidget(how_to_draw_my_plot, figsize=(5, 5))
ra = Range(min=0.0, max=2.0, step=0.1, value=plot.prop('amp'))
rf = Range(min=0.1, max=20.0, step=0.1, value=plot.prop('freq'))
show([plot, [show.style(textAlign='right'), 'Amp', ra,
             show.style(textAlign='right'),  'Freq', rf]])
```

This code shows the plot in a layout with two sliders.  If you later
execute the code `plot.freq = 5.0`, the plot will update live, in-place,
to show the new curve, and the freq slider will also move to 5.  And
of course, dragging the slider will also change the values live.

The [labwidget source code](https://github.com/davidbau/baukit/blob/main/baukit/labwidget.py) has much more detail.

## Online statistics library

`Covariance`, `Mean`, `Quantile`, `TopK`, and other data summarization
methods are provided as online, gpu-optimized algorithms.

```
from baukit import Quantile, Topk, CombinedStat, tally
cs = CombinedStat(
    qc=Quantile(),
    tk=TopK(),
)
ds = MyDataset()
# Loads from my_stats.npz if already computed.
for [batch] in tally(cs, ds, cache='my_stats.npz', batch_size=50):
    batch.cuda()
    # Assumes dim=0 is the sampling axis; stats are per dim=1 feature.
    stat.add(batch)
cs.to_('cpu')
median = cs.qc.quantile(0.5)
top_values, top_indexes = cs.tk.topk(10)
```

The [runningstats source code](https://github.com/davidbau/baukit/blob/main/baukit/runningstats.py) shows other things you can do.

## Improved basic dataset objects

`ImageFolderSet` is faster and provides more features than
pytorch `ImageFolder` including the ability to gather multiple
streams of parallel data tensors (such as segmentations and images).

`TokenizedDataset` tokenizes text through a provided tokenizer,
producing dictionaries designed to feed directly into `huggingface`
language models.  It works with `length_collation` for creating
uniform-length batches for fast training and inference.

## Batch job utilities

`pbar` is a more readable progress bar utility wrapper around `tqdm`
that simplifies the display of progress status strings during a
long progress operation; it also provides a way for a caller to
slience progress output.

`reserve_dir` reserves a directory for results of a job and grabs a lock
so that other proceses running `reserve_dir` will not do the same job.
This allows very simple batch parallelism: just run many processes
that run all the jobs, and each job will only be done once.

`WorkerPool` simplifies creation of worker threads for consuming output
data; this can dramatically speed up writing of many output files
and is the output analogue of the torch DataLoader utility for inputs.
