import time
import glob
import os.path as pth
import warnings
import pickle
import imageio
import cv2
import numpy as np
from PIL import Image


class Normalizer:
    def __init__(self, image, mask=None):
        assert (image.ndim >= 2)
        if mask is None:
            self.min_v = np.min(image, axis=(0, 1))
            self.max_v = np.max(image, axis=(0, 1))
        else:
            masked_image = image[mask]
            self.min_v = np.min(masked_image, axis=0)
            self.max_v = np.max(masked_image, axis=0)

    def normalize(self, image, mask=None):
        if mask is None:
            return (image - self.min_v) / (self.max_v - self.min_v)
        else:
            out = image.copy()
            out[mask] = (out[mask] - self.min_v) / (self.max_v - self.min_v)
            return out

    def denormalize(self, image, mask=None):
        if mask is None:
            return image * (self.max_v - self.min_v) + self.min_v
        else:
            out = image.copy()
            out[mask] = out[mask] * (self.max_v - self.min_v) + self.min_v
            return out


def shift_image(img):
    return np.clip(img + 0.5, a_min=0, a_max=1)


# return mask maps as bool
def load_refined_masks(input_path, mask_maps, prefix):
    n_masks = mask_maps.shape[0]
    masks = []
    exist = []
    for i in range(n_masks):
        map_name = prefix + str(i) + '0.png'
        mask_filename = pth.join(input_path, map_name)
        if pth.exists(mask_filename) is False:
            warnings.warn(f'Missing refined mask map: {map_name}')
            masks.append(mask_maps[i])
            exist.append(False)
        else:
            mask = read_binary_image(mask_filename)
            masks.append(mask_maps[i] * mask)
            exist.append(True)
    masks = np.stack(masks, axis=0)
    return masks, exist


# return mask maps as float point (0, 1)
def load_procedural_masks(input_path, n_masks=0, target_size=None, suffix='_final_pptbf_binary', allow_missing=False):
    if n_masks == 0:
        file_list = glob.glob(pth.join(input_path, 'bmask*.png'))
        n_masks = len(file_list)

    if allow_missing:
        assert n_masks == 2

    masks = [None] * n_masks
    valid_index = []
    for i_mask in range(n_masks):
        sub_output_path = pth.join(input_path, f'bmask{i_mask}')
        mask_filename = pth.join(sub_output_path, f'bmask{i_mask}{suffix}.png')
        if pth.exists(mask_filename):
            mask = read_image(mask_filename)
            if mask.ndim == 3:
                warnings.warn(f'Procedural mask map {mask_filename} has channels more than 2. Take the first channel.')
                mask = mask[:, :, 0]
            if target_size is not None and target_size != mask.size:
                mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            masks[i_mask] = mask
            valid_index.append(i_mask)

    if (not allow_missing and len(valid_index) != n_masks) or (allow_missing and len(valid_index) == 0):
        raise RuntimeError("Cannot find enough procedural masks")

    if allow_missing:
        another = 1 - valid_index[0]
        masks[another] = 1 - masks[1 - another]

    masks = np.stack(masks, axis=0)
    return masks


def load_image(filename, target_size=None, resample=Image.BILINEAR):
    image = Image.open(filename)
    if target_size is not None:
        image = image.resize(target_size, resample=resample)
    if image.mode == 'I':
        image = np.asarray(image, dtype=np.float32) / 65535
    elif image.mode == '1':
        image = np.asarray(image, dtype=np.bool)
    elif image.mode == 'RGB' or image.mode == 'L':
        image = np.asarray(image, dtype=np.float32) / 255.0
    else:
        print(f'Unrecognized image mode {image.mode}')
        raise RuntimeError(f'Unrecognized image mode {image.mode}')

    return image


def save_image(image, filename):
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            img = Image.fromarray(image)
        else:
            img = Image.fromarray((np.clip(image, 0, 1) * 255.0).astype(np.uint8))
        img.save(filename)
    elif isinstance(image, Image.Image):
        image.save(filename)
    else:
        raise NotImplementedError(f'Unknown image type: {type(image)}')


def make_grid(images, n_row, gap=5):
    n_images = len(images)
    n_row = min(n_row, n_images)
    if n_images == 0:
        return None
    w, h = images[0].shape[:2]
    c = 3

    n_col = int(np.ceil(n_images / n_row))
    out = np.ones((w * n_col + gap * (n_col - 1), h * n_row + gap * (n_row - 1), c))
    for j in range(n_col):
        for i in range(n_row):
            if j * n_row + i < n_images:
                if images[j * n_row + i].ndim == 3:
                    out[(w + gap) * j:(w + gap) * j + w, (h + gap) * i:(h + gap) * i + h] = images[j * n_row + i]
                else:
                    out[(w + gap) * j:(w + gap) * j + w, (h + gap) * i:(h + gap) * i + h] = images[j * n_row + i][..., np.newaxis]
            else:
                return out
    return out


def read_image(filename: str):
    img = imageio.imread(filename)
    if img.dtype == np.float32:
        return img
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        raise RuntimeError('Unexpected image data type.')


def read_binary_image(filename: str):
    img = imageio.imread(filename)
    if img.dtype == np.uint8:
        img = img.astype(np.bool)
    else:
        raise RuntimeError('Unexpected image data type.')
    return img


def write_image(img, filename: str):
    extension = pth.splitext(filename)[1]
    if extension == '.exr':
        imageio.imwrite(filename, img)
    elif extension in ['.png', '.jpg', '.bmp']:
        if img.dtype == np.uint8:
            imageio.imwrite(filename, img)
        else:
            imageio.imwrite(filename, (img * 255.0).astype(np.uint8))
    else:
        raise RuntimeError(f'Unexpected image filename extension {extension}.')


def load_svbrdf_maps(data_path):
    albedo_filename = pth.join(data_path, 'albedo.png')
    height_filename = pth.join(data_path, 'normal.png')
    roughness_filename = pth.join(data_path, 'roughness.png')
    albedo = read_image(albedo_filename)
    height = read_image(height_filename)
    roughness = read_image(roughness_filename)
    return albedo, height, roughness


def record_losses(loss_graph, losses):
    for name, loss in losses.items():
        if name in loss_graph:
            loss_graph[name].append(loss)
        else:
            loss_graph[name] = [loss]


class Timer:
    def __init__(self):
        self.start_time = []

    def begin(self, output=''):
        if output != '':
            print(output)
        self.start_time.append(time.time())

    def end(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time.pop()
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)