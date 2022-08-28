'''
Variants of pytorch's ImageFolder for loading image datasets with more
information, such as parallel feature channels in separate files,
cached files with lists of filenames, etc.
'''

import os
import torch
import re
import random
import numpy
import itertools
import copy
import torch.utils.data as data
from torchvision.datasets.folder import default_loader as tv_default_loader
from PIL import Image
from collections import OrderedDict


def default_loader(filename):
    '''
    Handles both numpy files and image formats.
    '''
    try:
        if filename.endswith('.npy') or filename.endswith('.NPY'):
            return numpy.load(filename).view(ndarray)
        elif filename.endswith('.npz') or filename.endswith('.NPZ'):
            return numpy.load(filename)
        else:
            return tv_default_loader(filename)
    except Exception as err:
        raise OSError('Unable to load ' + filename + ': ' + str(err))

class ImageFolderSet(data.Dataset):
    """
    A data loader that generalizes torchvision.datasets.ImageFolder,
    addding the following features:
     - Classification is optional (and defaults off); it can
       load just a plain folder hierarchy of images.
     - It can skip the slow folder walk and quickly initialize
       by looking for an `index.txt` file that lists filenames.
     - It can load directories containing npy or npz files as
       well as image formats like png, jpg, gif.
     - It can collate parallel folders with matching filenames, e.g.,

            data_slice_1/park/004234.jpg
            data_slice_1/park/004236.jpg
            data_slice_1/park/004237.jpg

            data_slice_2/park/004234.png
            data_slice_2/park/004236.png
            data_slice_2/park/004237.png

       Parallel files like 004234.jpg and 004234.png will be
       loaded as part of the same dataset item.

    Constructor arguments:

    image_roots: a directory name, or a list of directory names.
        Each directory defines one of the data slices.
    transform (optional): a callable, or a list of callables,
        for preprocessing the images after they are loaded.
        If a list, there should be one transform per image root.
    stacker (optional): if provided, the stacker is called to
        combine the processed data items into a single tensor;
        otherwise they are left separate.
    classification: set to True to use folder names as
        classification labels (default False)
    identification: set to True to include a unique sequence
        number in the data identifying each image.
    normalize_filename: data will be collated if the filenames
        match, up to normalization.  The default normalization
        strips the filename extension, but this callable can
        specify a different filename normalization rule.
    size: if specified, truncates data set to this number
        of items.
    shuffle: if specified, shuffles the data set instead of
        sorting by filename.  Pass an integer to specify the
        deterministic pseudorandom shuffle order.
    lazy_init: set to False to force the image walk to
        happen during the constructor; otherwise it is
        done when first needed.
    """
    def __init__(self,
                 image_roots,
                 transform=None,
                 loader=default_loader,
                 stacker=None,
                 classification=False,
                 identification=False,
                 intersection=False,
                 filter_tuples=None,
                 normalize_filename=None,
                 size=None,
                 shuffle=None,
                 lazy_init=True):
        if isinstance(image_roots, str):
            image_roots = [image_roots]
        self.image_roots = image_roots
        if transform is not None and not hasattr(transform, '__iter__'):
            transform = [transform for _ in image_roots]
        self.transforms = transform
        self.stacker = stacker
        self.loader = loader
        self.identification = identification

        def do_lazy_init():
            self.images, self.classes, self.class_to_idx = (
                make_parallel_dataset(image_roots,
                                      classification=classification,
                                      intersection=intersection,
                                      filter_tuples=filter_tuples,
                                      normalize_fn=normalize_filename
                                      ))
            if len(self.images) == 0:
                raise RuntimeError("Found 0 images within: %s" % image_roots)
            if shuffle is not None:
                random.Random(shuffle).shuffle(self.images)
            if size is not None:
                self.images = self.images[:size]
            self._do_lazy_init = None
        if lazy_init:
            self._do_lazy_init = do_lazy_init
        else:
            do_lazy_init()

    def subset(self, indexes):
        '''
        Returns a subset of the current dataset, given by
        the set of specified indexes.
        '''
        if self._do_lazy_init is not None:
            self._do_lazy_init()
        # Copy over transforms and other settings.
        ds = ImageFolderSet(
            self.image_roots,
            transform=self.transforms,
            loader=self.loader,
            stacker=self.stacker,
            identification=self.identification,
            lazy_init=True)
        # Initialize the subset items directly.
        ds.images = [
            copy.deepcopy(self.images[i]) for i in indexes]
        ds.classes = self.classes
        ds.class_to_idx = self.class_to_idx
        ds._do_lazy_init = None
        return ds

    def __getattr__(self, attr):
        # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        if not attr.startswith('_') and self._do_lazy_init is not None:
            self._do_lazy_init()
            return getattr(self, attr)
        raise AttributeError()

    def __getitem__(self, index):
        return self.get_augmented(index, None)

    def get_augmented(self, index, transform_arg=None):
        if self._do_lazy_init is not None:
            self._do_lazy_init()
        paths = self.images[index]
        if self.classes is not None:
            classidx = paths[-1]
            paths = paths[:-1]
        sources = [self.loader(path) for path in paths]
        # Add a common shared state dict to allow random crops/flips to be
        # coordinated.
        shared_state = {}
        for s in sources:
            try:
                s.shared_state = shared_state
            except BaseException:
                pass
        if self.transforms is not None:
            if transform_arg is None:
                call_transform = lambda t, s: t(s) if t is not None else s
            else:
                call_transform = lambda t, s: t(s, transform_arg) if t is not None else s
            sources = [
                    call_transform(transform, source)
                    for source, transform
                    in itertools.zip_longest(sources, self.transforms)]
        if self.stacker is not None:
            sources = self.stacker(sources)
            if self.classes is None and not self.identification:
                return sources
            else:
                sources = [sources]
        if self.classes is not None:
            sources.append(classidx)
        if self.identification:
            sources.append(index)
        sources = tuple(sources)
        return sources

    def __len__(self):
        if self._do_lazy_init is not None:
            self._do_lazy_init()
        return len(self.images)


def is_npy_file(path):
    return (path.endswith('.npy') or path.endswith('.NPY') or
            path.endswith('.npz') or path.endswith('.NPZ'))


def is_image_file(path):
    return None is not re.search(r'\.(jpe?g|png)$', path, re.IGNORECASE)


def walk_image_files(rootdir):
    # Skip the walk if an index.txt file is found.
    for indexfile, basedir in [
            ('%s/index.txt' % rootdir, rootdir),
            ('%s.txt' % rootdir, os.path.dirname(rootdir))]:
        if os.path.isfile(indexfile):
            with open(indexfile) as f:
                result = sorted([
                    os.path.normpath(os.path.join(basedir, line.strip()))
                    for line in f.readlines()])
                return result
    result = []
    for dirname, _, fnames in sorted(os.walk(rootdir)):
        for fname in sorted(fnames):
            if is_image_file(fname) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result

def make_parallel_dataset(image_roots, classification=False,
        intersection=False, filter_tuples=None, normalize_fn=None):
    """
    Returns ([(img1, img2, clsid, id), (img1, img2, clsid, id)..],
             classes, class_to_idx)
    """
    image_roots = [os.path.expanduser(d) for d in image_roots]
    image_sets = OrderedDict()
    if normalize_fn is None:
        def normalize_fn(x): return os.path.splitext(x)[0]
    for j, root in enumerate(image_roots):
        for path in walk_image_files(root):
            key = normalize_fn(os.path.relpath(path, root))
            if key not in image_sets:
                image_sets[key] = []
            if not intersection and len(image_sets[key]) != j:
                raise RuntimeError('Images not parallel: '
                                   '{} missing from {}'.format(key, root))
            image_sets[key].append(path)
    if classification:
        classes = sorted(set([os.path.basename(os.path.dirname(k))
                              for k in image_sets.keys()]))
        class_to_idx = dict({k: v for v, k in enumerate(classes)})
        for k, v in image_sets.items():
            v.append(class_to_idx[os.path.basename(os.path.dirname(k))])
    else:
        classes, class_to_idx = None, None
    tuples = []
    for key, value in image_sets.items():
        if len(value) != (len(image_roots) + (1 if classification else 0)):
            if intersection:
                continue
            else:
                raise RuntimeError(
                    'Images not parallel: %s missing from one dir' % (key))
        value = tuple(value)
        if filter_tuples and not filter_tuples(value):
            continue
        tuples.append(value)
    return tuples, classes, class_to_idx


class NpzToTensor:
    """
    A data transformer for converting a loaded npz file to a pytorch
    tensor.  Since an npz file stores tensors under keys, a key can be
    specified.  Otherwise, the first key is dereferenced.
    """

    def __init__(self, key=None):
        self.key = key

    def __call__(self, data):
        key = self.key or next(iter(data))
        return torch.from_numpy(data[key])

def grayscale_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')


class ndarray(numpy.ndarray):
    '''
    Wrapper to make ndarrays into heap objects so that shared_state can
    be attached as an attribute.
    '''
    pass
