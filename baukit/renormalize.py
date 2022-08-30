import numpy
import torch
import PIL
import io
import base64
import re
from torchvision import transforms


def as_tensor(data, source='zc', target='zc'):
    renorm = renormalizer(source=source, target=target)
    return renorm(data)


def as_image(data, source='zc', target='byte'):
    if len(data.shape) == 4:
        return [as_image(d, source, target) for d in data]
    assert len(data.shape) == 3
    renorm = renormalizer(source=source, target=target)
    return PIL.Image.fromarray(renorm(data).
                               permute(1, 2, 0).cpu().numpy())


def as_url(data, source='zc', size=None):
    if isinstance(data, PIL.Image.Image):
        img = data
    else:
        img = as_image(data, source)
    if size is not None:
        img = img.resize(size, resample=PIL.Image.BILINEAR)
    buffered = io.BytesIO()
    img.save(buffered, format='png')
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return 'data:image/png;base64,%s' % (b64)


def from_image(im, target='zc', size=None):
    if isinstance(im, list):
        return torch.stack([from_image(one, target, size) for one in im])
    if im.format != 'RGB':
        im = im.convert('RGB')
    if size is not None:
        im = im.resize(size, resample=PIL.Image.BILINEAR)
    pt = transforms.functional.to_tensor(im)
    renorm = renormalizer(source='pt', target=target)
    return renorm(pt)


def from_url(url, target='zc', size=None):
    image_data = re.sub('^data:image/.+;base64,', '', url)
    im = PIL.Image.open(io.BytesIO(base64.b64decode(image_data)))
    if target == 'image' and size is None:
        return im
    return from_image(im, target, size=size)


def renormalizer(source='zc', target='zc'):
    '''
    Returns a function that imposes a standard normalization on
    the image data.  The returned renormalizer operates on either
    3d tensor (single image) or 4d tensor (image batch) data.
    The normalization target choices are:

        zc (default) - zero centered [-1..1]
        pt - pytorch [0..1]
        imagenet - zero mean, unit stdev imagenet stats (approx [-2.1...2.6])
        byte - as from an image file, [0..255]

    If a source is provided (a dataset or transform), then, the renormalizer
    first reverses any normalization found in the data source before
    imposing the specified normalization.  When no source is provided,
    the input data is assumed to be pytorch-normalized (range [0..1]).
    '''
    if isinstance(source, str):
        oldoffset, oldscale = OFFSET_SCALE[source]
    else:
        normalizer = find_normalizer(source)
        oldoffset, oldscale = (
            (normalizer.mean, normalizer.std) if normalizer is not None
            else OFFSET_SCALE['pt'])
    newoffset, newscale = (target if isinstance(target, tuple)
                           else OFFSET_SCALE[target])
    return Renormalizer(oldoffset, oldscale, newoffset, newscale,
                        tobyte=(target == 'byte'))


# Several commonly-seen image normalization schemes.
OFFSET_SCALE = dict(
    # pytorch default [0, 1]
    pt=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    # zero-centered [-1, 1]
    zc=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # zero-mean, unit-variance over empirical ImageNet sample
    imagenet=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # clip normalization; see
    # https://github.com/openai/CLIP/blob/c5478aac7b9e007a2659d36b57ebe148849e542a/clip/clip.py#L85
    clip=([0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    # zero-mean, 255 range over ImageNet sample
    imagenet_meanonly=([0.485, 0.456, 0.406],
                       [1.0 / 255, 1.0 / 255, 1.0 / 255]),
    # zero-mean, 255 range over Places sample
    places_meanonly=([0.475, 0.441, 0.408],
                     [1.0 / 255, 1.0 / 255, 1.0 / 255]),
    # byte encoding [0, 255] as in common image file formats
    byte=([0.0, 0.0, 0.0], [1.0 / 255, 1.0 / 255, 1.0 / 255]))

NORMALIZER = {k: transforms.Normalize(*OFFSET_SCALE[k]) for k in OFFSET_SCALE}


def find_normalizer(source=None):
    '''
    Crawl around the transforms attached to a dataset looking for a
    Normalize transform to return.
    '''
    if source is None:
        return None
    if isinstance(source, (transforms.Normalize, Renormalizer)):
        return source
    t = getattr(source, 'transform', None)
    if t is not None:
        return find_normalizer(t)
    ts = getattr(source, 'transforms', None)
    if ts is not None:
        for t in reversed(ts):
            result = find_normalizer(t)
            if result is not None:
                return result
    return None


class Renormalizer:
    def __init__(self, oldoffset, oldscale, newoffset, newscale, tobyte=False):
        self.mul = torch.from_numpy(
            numpy.array(oldscale) / numpy.array(newscale))
        self.add = torch.from_numpy(
            (numpy.array(oldoffset) - numpy.array(newoffset))
            / numpy.array(newscale))
        self.tobyte = tobyte
        # Store these away to allow the data to be renormalized again
        self.mean = newoffset
        self.std = newscale

    def __call__(self, data):
        mul, add = [d.to(data.device, data.dtype)
                    for d in [self.mul, self.add]]
        if data.ndimension() == 3:
            mul, add = [d[:, None, None] for d in [mul, add]]
        elif data.ndimension() == 4:
            mul, add = [d[None, :, None, None] for d in [mul, add]]
        result = data.mul(mul).add_(add)
        if self.tobyte:
            result = result.clamp(0, 255).byte()
        return result
