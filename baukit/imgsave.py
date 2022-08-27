# A utility for saving a large number of images quickly without
# blocking a single thread to wait for each individual image to save.

import os
import PIL
from .workerpool import WorkerBase, WorkerPool
from . import pbar

def save_image_set(img_array, filename_pattern,
                   sourcefile=None, num_workers=None, quality=99):
    '''
    Saves all the (PIL) images in the given array, using the
    given filename pattern (which should contain a `{0}` to get
    the index number of the image).  If the array is nested,
    then it will loop over the nested contents, and the filename
    pattern should have an additional '{1}', and so on, for each
    level of nesting.
    '''
    if sourcefile is not None:
        last_filename = expand_last_filename(img_array, filename_pattern)
        # Do nothing if the last file exists and is newer than the sourcefile
        if os.path.isfile(last_filename) and (os.path.getmtime(last_filename)
                                              >= os.path.getmtime(sourcefile)):
            pbar.descnext(None)
            return
    # Use multiple threads to write all the image files faster.
    pool = WorkerPool(worker=SaveImageWorker, num_workers=num_workers)
    for img, filename in pbar(
            all_items_and_filenames(img_array, filename_pattern),
            total=num_items(img_array)):
        pool.add(img, filename, quality)
    pool.join()


class SaveImageWorker(WorkerBase):
    '''
    Does the slow, single-threaded work of saving a single image to a file.
    '''
    def work(self, img, filename, quality):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename, optimize=True, quality=quality)


class SaveImagePool(WorkerPool):
    '''
    A pool of parallel image-saving workers.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, worker=SaveImageWorker, **kwargs)

def all_items_and_filenames(img_array, filename_pattern, index=()):
    '''
    Uses filename_pattern to assign a filename to every item in the
    img_array, which may be nested.  The pattern should have one or
    more {} which will be expanded to the array indexes, with one
    index for each level of nesting.
    '''
    for i, data in enumerate(img_array):
        inner_index = index + (i,)
        if PIL.Image.isImageType(data):
            yield data, (filename_pattern.format(*inner_index))
        else:
            for img, name in all_items_and_filenames(data, filename_pattern,
                                                     inner_index):
                yield img, name

def expand_last_filename(img_array, filename_pattern):
    '''
    Returns the last filename that would be returned by
    all_items_and_filenames; used to check the presence of the last
    file that would be written.
    '''
    index, data = (), img_array
    while not PIL.Image.isImageType(data):
        index += (len(data) - 1,)
        data = data[len(data) - 1]
    return filename_pattern.format(*index)


def num_items(img_array):
    num = 1
    while not PIL.Image.isImageType(img_array):
        num *= len(img_array)
        img_array = img_array[-1]
    return num
