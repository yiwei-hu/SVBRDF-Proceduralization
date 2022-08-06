import glob
import os

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms


def get_image_files(path):
    image_file_exts = ['*.jpg', '*.png']
    image_files = []
    for ext in image_file_exts:
        image_files.extend(glob.glob(os.path.join(path, ext)))

    return image_files


def save_image(image, filename, single_channel=True):
    if isinstance(image, np.ndarray):
        img = Image.fromarray((np.clip(image, 0, 1) * 255.0).astype(np.uint8))
        img.save(filename)
    elif isinstance(image, Image.Image):
        image.save(filename)
    elif isinstance(image, torch.Tensor):
        if single_channel:
            PILImage = transforms.ToPILImage()(image.cpu())
            PILImage.save(filename)
        else:
            vutils.save_image(image, filename)
    else:
        raise NotImplementedError("unknown image type: {}".format(type(image)))


def dump_features(features, filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.txt':
        np.savetxt(filename, features, fmt='%.7g')
    elif ext == '.npy':
        np.save(filename, features)
    else:
        raise NotImplementedError('unsupported file extension: {}'.format(ext))


def load_features(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.txt':
        return np.loadtxt(filename)
    elif ext == '.npy':
        return np.load(filename)
    else:
        raise NotImplementedError('unsupported file extension: {}'.format(ext))


def load_parameter_file(filename):
    with open(filename, 'r') as f:
        param = f.read().replace('\n', ' ').replace('\r', ' ')
        param = [p for p in param.split(' ') if p]
    param = [float(p) for p in param]
    return np.array([param])


def save_parameter_file(parameter, filename):
    with open(filename, 'w') as f:
        f.write(str(parameter[0]) + '\n')
        for i in range(1, len(parameter)):
            f.write(str(parameter[i]) + ' ')
