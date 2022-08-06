import os
import os.path as pth
import warnings
import numpy as np
import torch
import imageio

from utils import read_image, write_image
from optim.render import Renderer, TextureOps


class Visualizer:
    @staticmethod
    def render_images(data_path, normal_intensity, lp, cp=None, suffixes=('', '(gt)', '(raw)')):
        if cp is None:
            cp = lp
        is_height = normal_intensity != 0

        output_path = pth.join(data_path, 'renders')
        os.makedirs(output_path, exist_ok=True)
        for suffix in suffixes:
            albedo, height, roughness = Visualizer.load_svbrdf(data_path, suffix, is_height)
            res = albedo.shape[2]

            im_size = 9.728
            li = np.asarray([1500, 1500, 1500], dtype=np.float64)
            light_settings = {'lp': lp, 'cp': cp, 'im_size': im_size, 'li': li}

            renderer = Renderer(res, normal_intensity=normal_intensity, light_settings=light_settings)
            with torch.no_grad():
                render_targets = renderer.render(albedo, height, roughness, is_height=is_height, return_index=False)

            for i, render_target in enumerate(render_targets):
                im = TextureOps.tensor2numpy(render_target)
                write_image(im, pth.join(output_path, f'render{i}{suffix}.png'))

    @staticmethod
    def animate_lighting(input_path, output_path, suffix='', normal_intensity=1, r=42.6, phi=np.pi/12,
                         n_round=2, n_seconds=4, frame_per_seconds=24):
        # load svbrdf
        is_height = normal_intensity != 0
        albedo, height, roughness = Visualizer.load_svbrdf(input_path, suffix, is_height)
        res = albedo.shape[2]

        # compute light sequence
        n_frames = n_seconds*frame_per_seconds
        frames_per_round = int(n_frames / n_round)
        theta = np.linspace(0, 2*np.pi, num=frames_per_round, endpoint=False)
        theta = np.tile(theta, n_round)
        phi = np.ones_like(theta)*phi
        x = r*np.cos(theta)*np.sin(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(phi)

        lp = np.stack((y, x, z), axis=1).astype(np.float64)
        cp = np.zeros_like(lp)
        cp[:, 2] = r

        # init lighting
        im_size = 9.728
        li = np.asarray([1500, 1500, 1500], dtype=np.float64)
        light_settings = {'lp': lp, 'cp': cp, 'im_size': im_size, 'li': li}

        renderer = Renderer(res, normal_intensity=normal_intensity, light_settings=light_settings)
        with torch.no_grad():
            render_targets = renderer.render(albedo, height, roughness, is_height=is_height, return_index=False)

        Visualizer.save_video(render_targets, output_path, suffix)

    @staticmethod
    def save_video(images, out_dir, suffix):
        n_frame = len(images)
        video = imageio.get_writer(f'{out_dir}/render{suffix}.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
        for i in range(n_frame):
            im = TextureOps.tensor2numpy(images[i])
            im255 = (im * 255.0).astype(np.uint8)
            video.append_data(im255)
        video.close()

    @staticmethod
    def load_svbrdf(input_path, suffix, use_height):
        albedo = read_image(Visualizer.get_filename(input_path, f'albedo{suffix}'))

        if use_height:
            print('load height map')
            normal_filename = pth.join(input_path, f'height{suffix}.npy')
            if pth.exists(normal_filename):
                normal = np.load(normal_filename)
            else:
                warnings.warn("Failed to find raw data for height map")
                normal = read_image(Visualizer.get_filename(input_path, f'height{suffix}'))
        else:
            print('load normal map')
            normal = read_image(Visualizer.get_filename(input_path, f'normal{suffix}'))

        roughness = read_image(Visualizer.get_filename(input_path, f'roughness{suffix}'))

        albedo = TextureOps.numpy2tensor(albedo)
        normal = TextureOps.numpy2tensor(normal)
        roughness = TextureOps.numpy2tensor(roughness)

        return albedo, normal, roughness

    @staticmethod
    def get_filename(data_path, map_name, exts=('.png', '.jpg', '.jpeg', '.bmp')):
        for ext in exts:
            filename = pth.join(data_path, map_name + ext)
            if pth.exists(filename):
                return filename
        raise FileNotFoundError(data_path + ':' + map_name)


def get_static_lights(dataset_name):
    if dataset_name in ['ground', 'mosaic', 'stone_smalltile']:
        lp = np.asarray([[0, 0, 30], [0, 20, 20]], dtype=np.float64)
    elif dataset_name in ['metal', 'painted_plaster']:
        lp = np.asarray([[0, 0, 40], [0, 30, 30]], dtype=np.float64)
    else:
        warnings.warn('Failed to find predefined lights. Use default instead.')
        lp = np.asarray([[0, 0, 30], [0, 20, 20]], dtype=np.float64)

    return lp


def get_animated_lights(dataset_name):
    if dataset_name in ['ground', 'mosaic', 'stone_smalltile']:
        return 30, np.pi / 12
    elif dataset_name in ['metal', 'painted_plaster']:
        return 42.6, np.pi / 12
    else:
        warnings.warn('Failed to find predefined lights. Use default instead.')
        return 30, np.pi / 12


def sample_light_position(r, n, phi=np.pi/12):
    np.set_printoptions(precision=2, suppress=True)

    theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    phi = np.ones_like(theta) * phi
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    lp = np.stack((y, x, z), axis=1).astype(np.float32)

    for i in range(lp.shape[0]):
        print(f'{lp[i][0]:.4f}, {lp[i][1]:.4f}, {lp[i][2]:.4f}')