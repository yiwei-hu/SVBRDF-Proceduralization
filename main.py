import os.path as pth
import warnings
import numpy as np
import torch

from matting.interface import start_app
from multi_material import Synthesizer, Resynthesizer
from normal_intensity import get_normal_intensity
from visualizer import Visualizer, get_static_lights, get_animated_lights

warnings.simplefilter("default")

dataset_name = 'ground'  # ['ground', 'metal', 'mosaic', 'painted_plaster', 'stone_smalltile']
root_path = f'./samples/{dataset_name}'
normal_intensity = get_normal_intensity(dataset_name)

np.set_printoptions(suppress=False)
torch.set_printoptions(sci_mode=False)


def synthesize():
    np.set_printoptions(suppress=True)

    # create a synthesizer
    synthesizer = Synthesizer(root_path)

    # proceduralize structure (mask maps)
    synthesizer.proceduralize_mask_maps(ignore_exist=True)

    # proceduralize materials (generate noise models and build the graph hierarchy)
    synthesizer.proceduralize_material(preload=False)

    # differentiable reconstruction
    # please use visdom to visualize losses: python -m visdom.server -port 8000
    synthesizer.global_optimization(normal_intensity=normal_intensity, use_procedural_mask=False)

    visualize(synthesizer.save_path, normal_intensity)


def resynthesize():
    options = ['generate',
               'synthesis',
               'superres'
               ]
    size = 512

    model = Resynthesizer(root_path)

    # for rendering, if using height maps, normal_intensity should be manually rescaled with the resolution
    use_height = True
    normal = normal_intensity if use_height else 0
    for option in options:
        if option == 'generate':  # reproduce optimization results
            model.generate(normal_intensity=normal_intensity, use_procedural_mask=False)
            visualize(pth.join(root_path, 'results/synthesis_results/generate'), normal, suffices=('',))

        elif option == 'synthesis':  # material synthesis
            size_ = size
            model.synthesize(size=size_, normal_intensity=normal_intensity, trim=True)
            visualize(pth.join(root_path, 'results/synthesis_results/synthesis'), normal, suffices=('',))

        elif option == 'superres': # material super-resolution (x2)
            size_ = size * 2
            model.super_resolve(size=size_, normal_intensity=normal_intensity)
            visualize(pth.join(root_path, 'results/synthesis_results/super_resolve'), normal, suffices=('',))


def visualize(data_path, intensity, suffices=('', '(gt)', '(raw)')):
    Visualizer.render_images(data_path, intensity, get_static_lights(dataset_name), suffixes=suffices)
    r, phi = get_animated_lights(dataset_name)
    for suffix in suffices:
        Visualizer.animate_lighting(data_path, data_path, suffix=suffix, normal_intensity=intensity, r=r, phi=phi)


if __name__ == '__main__':
    start_app()
    # synthesize()
    # resynthesize()
