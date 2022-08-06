import os.path as pth
import numpy as np
from PIL import Image
import matlab.engine
from scipy import ndimage
import cv2
from sklearn.decomposition import PCA

from utils import read_binary_image, write_image


def mask_binarization(alpha):
    n_masks = alpha.shape[0]
    max_indices = np.argmax(alpha, axis=0)
    masks = np.transpose(np.eye(n_masks, dtype=bool)[max_indices], axes=(2, 0, 1))
    return masks


def save_masks_as_image(masks, output_path, prefix="mask"):
    assert (masks.ndim == 3)
    n_masks = masks.shape[0]
    for i in range(n_masks):
        mask = masks[i]
        save_path = pth.join(output_path, prefix + str(i) + '.png')
        print(f'saving at {save_path}')
        write_image(mask, save_path)


def load_masks(input_path, n_masks, prefix="mask"):
    max_num_masks = 20
    masks = []
    for i in range(max_num_masks):
        mask_filename = pth.join(input_path, prefix + str(i) + '.png')
        if pth.exists(mask_filename) is False:
            break
        mask = read_binary_image(mask_filename)
        masks.append(mask)
    if len(masks) == 0:
        raise RuntimeError(f'Failed to preload mask map: {prefix}')
    if n_masks > 0 and n_masks != len(masks):
        raise RuntimeError(f'Incorrect number of masks: expect {n_masks} but got {len(masks)}')
    masks = np.stack(masks, axis=0)
    return masks


def center_crop_resize(img, shape, resample=Image.BICUBIC):
    assert (shape[0] == shape[1])
    if isinstance(img, Image.Image):
        w = img.size[0]
        h = img.size[1]
        cropped_img = center_crop(img, shape=(min(w, h), min(w, h)))
        return cropped_img.resize(shape, resample=resample)
    else:
        raise NotImplementedError("Unsupported image type")


def center_crop(img, shape):
    if isinstance(img, np.ndarray):
        w = img.shape[0]
        h = img.shape[1]
        if w < shape[0] or h < shape[1]:
            shape = (min(w, h), min(w, h))

        x = (shape[0] - w) // 2
        y = (shape[1] - h) // 2

        return img[x:x + shape[0], y:y + shape[1]]
    elif isinstance(img, Image.Image):
        w = img.size[0]
        h = img.size[1]
        if w < shape[0] or h < shape[1]:
            shape = (min(w, h), min(w, h))

        x = (w - shape[0]) // 2
        y = (h - shape[1]) // 2

        return img.crop((x, y, x + shape[0], y + shape[1]))
    else:
        raise NotImplementedError(f'Unknown image type: {type(img)}')


def normalize(x):
    min_ = np.min(x)
    max_ = np.max(x)
    return (x - min_) / (max_ - min_)


def expand_dim_to_3(image):
    if image.ndim == 3:
        return image
    else:
        return np.expand_dims(image, axis=2)


def normal2height(normal, intensity=10, opengl=True):
    print(f'Is OpenGL normal: {opengl}')
    eng = matlab.engine.start_matlab()
    eng.cd('matting')
    height_ml = eng.normal2height(matlab.double(normal.tolist()), float(intensity), opengl)
    height = np.array(height_ml._data).reshape(height_ml.size, order='F')
    eng.quit()
    return height


def smooth_height_map(height):
    # smooth
    smoothed = height.copy()
    for i in range(2):
        smoothed = ndimage.median_filter(smoothed, size=5)

    # truncate peaks
    hi = np.percentile(smoothed.flatten(), 99.9)
    lo = np.percentile(smoothed.flatten(), 0.1)
    smoothed[smoothed > hi] = hi
    smoothed[smoothed < lo] = lo

    return smoothed


##################################################
# helper function for instance-based segmentation
##################################################


def reduce_dim(features, target_dim, normalized):
    n_samples, n_features = features.shape

    reduced = PCA(n_components=min(target_dim, n_samples, n_features)).fit_transform(features)

    return reduced if not normalized else normalize(reduced)


def find_bounding_box(mask, padding=0):
    w, h = mask.shape
    points_x, points_y = np.nonzero(mask)
    points = np.stack([points_x, points_y], axis=1)
    x, y, rw, rh = cv2.boundingRect(points)

    sx = max(0, x - padding)
    ex = min(w, x + rw + padding)
    sy = max(0, y - padding)
    ey = min(h, y + rh + padding)

    return sx, sy, ex, ey