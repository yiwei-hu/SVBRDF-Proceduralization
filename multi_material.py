import os
import os.path as pth
import shutil
import numpy as np
import torch
from skimage.measure import label as labeling
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import PPTBF.PPTBFModel as PPTBFModel
from PPTBF.PPTBF import PPTBFFitter
from PPTBF.utils import load_parameter_file, save_parameter_file

from noise.noise import decompose
import noise.procedures as procedures
from optim.material_graph import MaterialGraph
from optim.optimizer import MaterialGraphOptimizer

from utils import Timer, load_procedural_masks, load_svbrdf_maps, load_refined_masks
from utils import write_image, load_pickle, save_pickle

from config import Configurations


# a hierarchical material class for user interface
class MultiMaterial:
    class Layer:
        def __init__(self, prev=None, mask=None):
            self.prev = prev
            self.mask = mask
            self.next = None
            self.valid = False
            self.is_instance = False  # indicate whether it will modeled by probability
            self.is_inpainted = False  # indicate whether current layer is inpainted

    def __init__(self, svbrdf):
        self.albedo, self.normal, self.roughness = svbrdf
        self.root = self.Layer()
        self.current = self.root

    def next_layer(self, index):
        if self.current.valid is False:
            raise RuntimeError("Mask has been outdated")
        if self.current.mask.shape[0] <= index:
            raise RuntimeError("Index Out of Range")
        if self.current.next[index] is None:
            self.current.next[index] = self.Layer(self.current)
        self.current = self.current.next[index]

    def prev_layer(self):
        self.current = self.current.prev

    def reset(self):
        self.current = self.root

    def update_mask(self, mask, is_instance=False):
        self.current.mask = mask
        self.current.next = [None]*mask.shape[0]
        self.current.is_instance = is_instance
        self.set_valid(True)

    def invalidate_children(self, layer):
        if layer is None:
            return
        layer.valid = False
        if layer.next is not None:
            for child in layer.next:
                self.invalidate_children(child)

    def use_inpaint(self, is_inpainted):
        self.current.is_inpainted = is_inpainted

    def check_valid(self):
        return self.current.valid

    def set_valid(self, valid):
        self.current.valid = valid
        if not valid:
            self.invalidate_children(self.current)

    def is_modeled_by_instance(self):
        return self.current.is_instance

    def is_inpainted(self):
        return self.current.is_inpainted


# Procedural nodes in pixel map representation
class ProceduralComponents:
    def __init__(self, is_leaf, mask_maps=(None, None), svbrdf=(None, None, None, None, None, None)):
        self.is_leaf = is_leaf

        # non-leaf-node
        self.procedural_mask_maps = mask_maps[0]
        self.mask_maps = mask_maps[1]

        # leaf_node
        self.albedo_maps = svbrdf[0]
        self.base_albedo = svbrdf[1]
        self.height_maps = svbrdf[2]
        self.base_height = svbrdf[3]
        self.roughness_maps = svbrdf[4]
        self.base_roughness = svbrdf[5]

        # next layer
        self.next = [None]*self.procedural_mask_maps.shape[0] if self.procedural_mask_maps is not None else None


def is_leaf(layer):
    if layer is None or layer.mask is None or layer.valid is False:
        return True
    else:
        return False


class MaskProceduralizer:
    def __init__(self, ignore_exist):
        self.ignore_exist = ignore_exist
        self.pptbf_fitter = PPTBFFitter(Configurations.get_pptbf_fitting_params())

    @staticmethod
    def check_ppbtf_mask(data_path):
        mask0 = pth.join(data_path, 'bmask0/bmask0_final_pptbf_binary.png')
        mask1 = pth.join(data_path, 'bmask1/bmask1_final_pptbf_binary.png')
        if pth.exists(mask0) or pth.exists(mask1):
            print(f'Found existing procedural maps at {data_path}')
            return True
        else:
            return False

    def proceduralize(self, layer, data_path, depth):
        if is_leaf(layer):
            return

        if not layer.is_instance:
            if depth == 0 or layer.is_inpainted:  # root node or the mask has been inpainted
                mask_file = None
            else:
                mask_file = pth.join(data_path, 'prev_mask.png')

            if not (self.ignore_exist and self.check_ppbtf_mask(data_path)):
                target_files = [pth.join(data_path, f'{mask_name}.png') for mask_name in ['bmask0', 'bmask1']]
                losses = self.pptbf_fitter.fit(target_files, mask_file)
                # select the PPTBF mask with the minimal loss
                if losses[0] > losses[1]:
                    shutil.rmtree(pth.join(data_path, 'bmask0'))
                else:
                    shutil.rmtree(pth.join(data_path, 'bmask1'))
        else:
            random_sample_instances(layer.mask, data_path)

        for i, next_layer in enumerate(layer.next):
            self.proceduralize(next_layer, pth.join(data_path, f'layer{i}'), depth + 1)


class AppearanceProceduralizer:
    def proceduralize(self, layer, data_path, prev_svbrdf, prev_masks, depth):
        if layer is None:
            return

        output_path = pth.join(data_path, 'noises')
        albedo_save_path = pth.join(output_path, 'albedo')
        height_save_path = pth.join(output_path, 'height')
        roughness_save_path = pth.join(output_path, 'roughness')
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(albedo_save_path, exist_ok=True)
        os.makedirs(height_save_path, exist_ok=True)
        os.makedirs(roughness_save_path, exist_ok=True)

        is_leaf_node = []
        for next_layer in layer.next:
            is_leaf_node.append(is_leaf(next_layer))

        mask_maps = layer.mask

        # We need to re-label them and notice that the relabeled mask is used for optimization.
        # TODO: current re-labeling is random resampling while it can refined by precise region growth.
        if layer.is_instance:
            idx = int(pth.basename(data_path)[5:])
            gt_mask_maps = assign_labels(layer.prev.mask[idx], compute_probability(mask_maps)).astype(np.float) / 255.0
        # if current layer is decomposed by a inpainted image. We need to mask the mask maps by its parent mask
        elif layer.is_inpainted:
            idx = int(pth.basename(data_path)[5:])
            gt_mask_maps = (mask_maps * prev_masks[1][idx])
        else:
            gt_mask_maps = mask_maps.astype(np.float)

        # load procedural mask maps
        if layer.is_instance:
            procedural_mask_maps = load_procedural_masks(data_path, n_masks=mask_maps.shape[0], target_size=mask_maps.shape[1:], suffix='')
        else:
            procedural_mask_maps = load_procedural_masks(data_path, n_masks=mask_maps.shape[0], target_size=mask_maps.shape[1:], suffix='_final_pptbf_binary', allow_missing=True)

            # We manually mask the procedural map by its parent mask
            if prev_masks is not None:
                idx = int(pth.basename(data_path)[5:])
                procedural_mask_maps *= prev_masks[0][idx]

        # if this layer was inpainted, load the inpainted material maps from the disk
        if depth == 0 or layer.is_inpainted:
            albedo, height, roughness = load_svbrdf_maps(data_path)
        else:
            albedo, height, roughness = prev_svbrdf

        # proceduralize material maps
        albedo_maps, base_albedo = svbrdf_decomposition(albedo, mask_maps, data_path, albedo_save_path, is_leaf_node)
        height_maps, base_height = svbrdf_decomposition(height, mask_maps, data_path, height_save_path, is_leaf_node)
        roughness_maps, base_roughness = svbrdf_decomposition(roughness, mask_maps, data_path, roughness_save_path, is_leaf_node)

        procedural = ProceduralComponents(False, mask_maps=(procedural_mask_maps, gt_mask_maps))
        for i, next_layer in enumerate(layer.next):
            if is_leaf_node[i]:
                svbrdf = (albedo_maps[i], base_albedo[i], height_maps[i], base_height[i], roughness_maps[i], base_roughness[i])
                procedural.next[i] = ProceduralComponents(True, svbrdf=svbrdf)
            else:
                procedural.next[i] = self.proceduralize(next_layer, pth.join(data_path, f'layer{i}'), (albedo, height, roughness),
                                                        (procedural_mask_maps, gt_mask_maps), depth + 1)

        return procedural


class Synthesizer:
    def __init__(self, data_path, output_folder='results'):
        self.data_path = pth.join(data_path, output_folder)
        shutil.copyfile('config.py', pth.join(self.data_path, 'config.py'))
        self.material = load_pickle(pth.join(self.data_path, 'material.pkl'))

        if self.material is None:
            raise RuntimeError('Failed to load material from data path {}'.format(self.data_path))

        if self.material.albedo.dtype == np.uint8:  # old format
            self.albedo = self.material.albedo / 255.0
            self.height = self.material.normal[:, :, 0] / 255.0
            self.roughness = self.material.roughness[:, :, 0] / 255.0
        else:
            self.albedo = self.material.albedo
            self.height = self.material.normal
            self.roughness = self.material.roughness

        self.procedural_components = None
        self.save_path = pth.join(self.data_path, 'synthesis_results')
        os.makedirs(self.save_path, exist_ok=True)

    def proceduralize_mask_maps(self, ignore_exist):
        proceduralizer = MaskProceduralizer(ignore_exist)
        proceduralizer.proceduralize(self.material.root, self.data_path, 0)

    def proceduralize_material(self, preload):
        file_path = pth.join(self.data_path, 'components.pkl')
        if preload and pth.exists(file_path):
            procedural_components = load_pickle(file_path)
        else:
            proceduralizer = AppearanceProceduralizer()
            procedural_components = proceduralizer.proceduralize(self.material.root, self.data_path, None, None, 0)
            save_pickle(procedural_components, file_path)

        self.procedural_components = procedural_components

    # differentiable reconstruction by end-to-end material graph optimization
    def global_optimization(self, normal_intensity, use_procedural_mask=True):
        if self.procedural_components is None:
            self.proceduralize_material(preload=True)

        albedo, height, roughness = load_svbrdf_maps(self.data_path)
        target_images = {'albedo': albedo,
                         'height': height,
                         'roughness': roughness}

        cfg = Configurations.get_optimization_parameters()

        model = MaterialGraphOptimizer(self.procedural_components, target_images, normal_intensity, cfg, self.save_path,
                                       use_procedural_mask=use_procedural_mask)

        model.optimize()

        model.visualize()


class Resynthesizer(Synthesizer):
    def __init__(self, data_path):
        super(Resynthesizer, self).__init__(data_path)
        self.size = 0
        self.rescale = 1  # rescale PPTBF
        self.trim = False
        self.cfg = Configurations.get_optimization_parameters()

    # simply reproduce the optimization results
    def generate(self, normal_intensity, use_procedural_mask=True):
        procedural_components = load_pickle(pth.join(self.data_path, 'components.pkl'))

        graph = MaterialGraph(procedural_components, self.cfg, None, use_procedural_mask=use_procedural_mask)
        graph.load(pth.join(self.save_path, 'graph.torch'))
        graph.eval()

        rescale = None if self.cfg.target_size is None else 512 / self.cfg.target_size[0]
        if rescale is not None:
            with torch.no_grad():
                graph.scale_params(rescale)
            normal_intensity *= rescale

        output_path = pth.join(self.save_path, 'generate')
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            svbrdf = graph()
        MaterialGraphOptimizer.save_material_maps(svbrdf, normal_intensity, output_path, suffix='')

        with torch.no_grad():
            svbrdf_raw = graph.naive()
        MaterialGraphOptimizer.save_material_maps(svbrdf_raw, normal_intensity, output_path, suffix='(raw)')

        # copy ground truth
        shutil.copyfile(pth.join(self.save_path, 'albedo(gt).png'), pth.join(output_path, 'albedo(gt).png'))
        shutil.copyfile(pth.join(self.save_path, 'height(gt).npy'), pth.join(output_path, 'height(gt).npy'))
        shutil.copyfile(pth.join(self.save_path, 'height(gt).png'), pth.join(output_path, 'height(gt).png'))
        shutil.copyfile(pth.join(self.save_path, 'normal(gt).png'), pth.join(output_path, 'normal(gt).png'))
        shutil.copyfile(pth.join(self.save_path, 'roughness(gt).png'), pth.join(output_path, 'roughness(gt).png'))

        label_map = LabelMap()
        label_map.labeling_mask_maps(self.material.root, self.data_path, output_path)

    # synthesize a higher resolution material map
    def synthesize(self, size, normal_intensity, trim=False):
        self.size = size
        self.rescale = 1
        self.trim = trim

        rescale = None if self.cfg.target_size is None else 512 / self.cfg.target_size[0]

        timer = Timer()

        timer.begin("Synthesizing mask maps")
        pptbf_generator = PPTBFModel.PPTBFGenerator()
        self.synthesize_mask_maps(self.material.root, self.data_path, pptbf_generator)
        timer.end('Mask map synthesis finished in ')

        timer.begin("Synthesizing noise maps")
        procedural_components = self.regenerate_procedural_components(self.material.root, self.data_path)
        timer.end('Noise map synthesis finished in ')

        graph = MaterialGraph(procedural_components, self.cfg, None)
        graph.load(pth.join(self.save_path, 'graph.torch'))
        graph.eval()

        if rescale is not None:
            with torch.no_grad():
                graph.scale_params(rescale)
            normal_intensity *= rescale

        output_path = pth.join(self.save_path, 'synthesis')
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            svbrdf = graph()
        MaterialGraphOptimizer.save_material_maps(svbrdf, normal_intensity, output_path, suffix='')

        with torch.no_grad():
            svbrdf_raw = graph.naive()
        MaterialGraphOptimizer.save_material_maps(svbrdf_raw, normal_intensity, output_path, suffix='(raw)')

    def synthesize_mask_maps(self, layer, data_path, pptbf_generator):
        if is_leaf(layer):
            return

        # find PPTBF mask map
        which_mask = 'bmask0'
        if not pth.exists(pth.join(data_path, which_mask)):
            which_mask = 'bmask1'
        if not pth.exists(pth.join(data_path, which_mask)):
            raise FileNotFoundError('PPTBF parameters are missing')

        if not layer.is_instance:
            mask_params_filename_origin = pth.join(data_path, which_mask, f'{which_mask}_final_pptbf_params.txt')
            mask_params_filename_ext = pth.join(data_path, which_mask, f'{which_mask}_ext_pptbf_params.txt')
            params = load_parameter_file(mask_params_filename_origin)[0]
            if self.rescale != 1:
                params[3] = int(params[3] * self.rescale)

            # manually set distortion to zero. It can prevent weird distortion when synthesizing high-resolution masks
            if self.trim:
                params[6] = 0

            save_parameter_file(params, mask_params_filename_ext)
            pptbf_generator.generate(width=self.size, height=self.size, filename=mask_params_filename_ext)
        else:
            random_sample_instances(layer.mask, data_path, ignore_exist=False, input_suffix='_ext_pptbf_binary', output_suffix='_ext')

        for i, next_layer in enumerate(layer.next):
            self.synthesize_mask_maps(next_layer, pth.join(data_path, f'layer{i}'), pptbf_generator)

    def regenerate_procedural_components(self, layer, data_path, prev_mask_maps=None):
        is_leaf_node = []
        for i, next_layer in enumerate(layer.next):
            is_leaf_node.append(is_leaf(next_layer))

        n_sub_materials = layer.mask.shape[0]

        # load procedural mask maps
        if layer.is_instance:
            procedural_mask_maps = load_procedural_masks(data_path, n_masks=n_sub_materials, suffix='_ext')
        else:
            procedural_mask_maps = load_procedural_masks(data_path, n_masks=n_sub_materials, suffix='_ext_pptbf_binary', allow_missing=True)
            # We manually mask the procedural map by its parent mask
            if prev_mask_maps is not None:
                idx = int(pth.basename(data_path)[5:])
                procedural_mask_maps *= prev_mask_maps[idx]

        albedo_data_path = pth.join(data_path, 'noises/albedo')
        height_data_path = pth.join(data_path, 'noises/height')
        roughness_data_path = pth.join(data_path, 'noises/roughness')

        albedo_maps, albedo_color = self.resynthesize_noise_maps(albedo_data_path, n_sub_materials, self.size, is_leaf_node)
        height_maps, base_height = self.resynthesize_noise_maps(height_data_path, n_sub_materials, self.size, is_leaf_node)
        roughness_maps, base_roughness = self.resynthesize_noise_maps(roughness_data_path, n_sub_materials, self.size, is_leaf_node)

        procedural = ProceduralComponents(False, mask_maps=(procedural_mask_maps, None))
        for i, next_layer in enumerate(layer.next):
            if is_leaf_node[i]:
                svbrdf = (albedo_maps[i], albedo_color[i], height_maps[i], base_height[i], roughness_maps[i], base_roughness[i])
                procedural.next[i] = ProceduralComponents(True, svbrdf=svbrdf)
            else:
                procedural.next[i] = self.regenerate_procedural_components(next_layer, pth.join(data_path, f'layer{i}'), procedural_mask_maps)

        return procedural

    @staticmethod
    def resynthesize_noise_maps(save_path, n_sub_materials, size, valid):
        multilevel_syn_noises = []
        base_colors = []
        for i in range(n_sub_materials):
            if valid[i]:
                model_name = pth.join(save_path, 'model{}.pkl'.format(i))
                procedural_maps = load_pickle(model_name)
                syn_noises, base_color = procedural_maps(size)
                multilevel_syn_noises.append(syn_noises)
                base_colors.append(base_color)
            else:
                multilevel_syn_noises.append(None)
                base_colors.append(None)
        return multilevel_syn_noises, base_colors

    # super-resolve the material
    def super_resolve(self, size, normal_intensity):
        self.size = size
        self.rescale = size / 512
        self.trim = False

        rescale = self.rescale * (1 if self.cfg.target_size is None else 512 / self.cfg.target_size[0])

        timer = Timer()

        timer.begin("Synthesizing mask maps")
        pptbf_generator = PPTBFModel.PPTBFGenerator()
        self.synthesize_mask_maps(self.material.root, self.data_path, pptbf_generator)
        timer.end('Mask map synthesis finished in ')

        timer.begin("Synthesizing noise maps")
        procedures.upscaling_psd = True
        procedural_components = self.regenerate_procedural_components(self.material.root, self.data_path)
        procedures.upscaling_psd = False
        timer.end('Noise map synthesis finished in ')

        graph = MaterialGraph(procedural_components, self.cfg, None)
        graph.load(pth.join(self.save_path, 'graph.torch'))
        graph.eval()

        with torch.no_grad():
            graph.scale_params(rescale)
        normal_intensity *= rescale

        output_path = pth.join(self.save_path, 'super_resolve')
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            svbrdf = graph()
        MaterialGraphOptimizer.save_material_maps(svbrdf, normal_intensity, output_path, suffix='')
        with torch.no_grad():
            svbrdf_raw = graph.naive()
        MaterialGraphOptimizer.save_material_maps(svbrdf_raw, normal_intensity, output_path, suffix='(raw)')


# hierarchically generate label maps
class LabelMap:
    def __init__(self):
        self.i_label = 1

    def generate_label_maps_(self, layer, data_path, depth):
        if is_leaf(layer):
            return
        mask_maps = layer.mask

        # load procedural mask maps
        if layer.is_instance:
            proc_mask_maps = load_procedural_masks(data_path, n_masks=mask_maps.shape[0], target_size=mask_maps.shape[1:], suffix='')
        else:
            proc_mask_maps = load_procedural_masks(data_path, n_masks=mask_maps.shape[0], target_size=mask_maps.shape[1:], suffix='_final_pptbf_binary', allow_missing=True)

        proc_mask_maps = proc_mask_maps.astype(np.bool)
        label_map = np.zeros(mask_maps.shape[1:], dtype=np.uint8)
        proc_label_map = np.zeros(proc_mask_maps.shape[1:], dtype=np.uint8)
        for i, next_layer in enumerate(layer.next):
            if is_leaf(next_layer):
                label_map[mask_maps[i]] = self.i_label
                proc_label_map[proc_mask_maps[i]] = self.i_label
                self.i_label += 1
            else:
                label_map_, proc_label_map_ = self.generate_label_maps_(next_layer, pth.join(data_path, f'layer{i}'), depth+1)
                label_map[mask_maps[i]] = label_map_[mask_maps[i]]
                proc_label_map[proc_mask_maps[i]] = proc_label_map_[proc_mask_maps[i]]
        return label_map, proc_label_map

    def generate_label_maps(self, material, data_path):
        self.i_label = 1
        label_map, proc_label_map = self.generate_label_maps_(material, data_path, 0)
        return label_map, proc_label_map, self.i_label - 1

    def labeling_mask_maps(self, material, data_path, output_path):
        label_map, proc_label_map, n_labels = self.generate_label_maps(material, data_path)

        color_map = cm.get_cmap('viridis', n_labels)
        label_map_rgb = label2rgb(label_map, bg_label=0, colors=color_map.colors)
        proc_label_map_rgb = label2rgb(proc_label_map, bg_label=0, colors=color_map.colors)

        os.makedirs(output_path, exist_ok=True)
        write_image(label_map_rgb, pth.join(output_path, 'label_map(gt).png'))
        write_image(proc_label_map_rgb, pth.join(output_path, 'label_map.png'))


#######################################################################################################################

# generate procedural mask maps by random sampling
def random_sample_instances(mask_maps, data_path, ignore_exist=True, prob=None,
                            input_suffix='_final_pptbf_binary', output_suffix=''):
    if ignore_exist and pth.exists(pth.join(data_path, 'bmask0')):
        print(f'Found existing procedural maps at {data_path}')
        return

    # compute probability based on segmentation
    if prob is None:
        prob = compute_probability(mask_maps)

    # load procedural mask maps from previous layer
    prev_data_path = pth.dirname(data_path)
    idx = int(pth.basename(data_path)[5:])  # layer{idx}
    try:
        mask = load_procedural_masks(prev_data_path, suffix=input_suffix, allow_missing=True)
    except FileNotFoundError:
        raise RuntimeError("Failed to locate procedural mask maps")

    mask = (mask[idx] * 255.0).astype(np.uint8)
    assert (len(np.unique(mask)) == 2)

    # assign instances to mask maps based on probability
    labeled_mask_maps = assign_labels(mask, prob)

    for i in range(mask_maps.shape[0]):
        output_path = pth.join(data_path, f'bmask{i}')
        os.makedirs(output_path, exist_ok=True)
        write_image(labeled_mask_maps[i], pth.join(output_path, f'bmask{i}{output_suffix}.png'))


# compute instance distributions based on segmented mask maps
def compute_probability(mask_maps):
    n_mask = mask_maps.shape[0]
    tot = 0
    nums = []
    for i in range(n_mask):
        _, num = labeling(mask_maps[i], return_num=True)
        nums.append(num)
        tot += num
    prob = []
    for i in range(n_mask):
        prob.append(nums[i] / tot)
    return prob


# # assign instances to mask maps based on probability
def assign_labels(mask_map,  prob):
    assert mask_map.ndim == 2
    # counting instances on procedural mask maps
    labeled_mask, num = labeling(mask_map, return_num=True)

    # assign labels to each instance based on probability
    assigned_labels = np.random.choice(np.arange(0, len(prob)), size=(num,), p=prob)
    labeled_mask_maps = np.zeros((len(prob), *mask_map.shape), dtype=np.uint8)
    for i, label in enumerate(assigned_labels):
        idx = np.where(labeled_mask == i + 1)
        labeled_mask_maps[label, idx[0], idx[1]] = 255

    return labeled_mask_maps


# proceduralize material appearance using a hierarchical noise model
def svbrdf_decomposition(image, mask_maps, data_path, save_path, valid_mask=None, enable_preload=True):
    if valid_mask is None:
        valid_mask = [True]*mask_maps.shape[0]

    name_of_map = pth.basename(save_path)
    cfg = Configurations.get_noise_model_params(name_of_map)

    name_of_map = name_of_map.capitalize()
    if name_of_map == 'Height':
        name_of_map = 'Normal' # TODO: should unify the naming
    refined_mask_maps, refined_masks_exist = load_refined_masks(data_path, mask_maps, prefix=f'fmask{name_of_map}')

    assert(mask_maps.shape[0] == refined_mask_maps.shape[0])

    n_masks = refined_mask_maps.shape[0]
    multilevel_syn_noises = []
    base_colors = []
    for i_mask in range(n_masks):
        if valid_mask[i_mask] is True:
            precomputed = pth.join(save_path, 'result{}.pkl'.format(i_mask))
            model_name = pth.join(save_path, 'model{}.pkl'.format(i_mask))
            if pth.exists(precomputed) and enable_preload:
                dat = load_pickle(precomputed)
                syn_noises = dat['syn_noises']
                base_color = dat['base_color']
            else:
                output_sub_path = pth.join(save_path, "mask" + str(i_mask))
                os.makedirs(output_sub_path, exist_ok=True)

                syn_noises, base_color, noise_models = decompose(image, refined_mask_maps[i_mask], cfg,
                                                                 is_mask_refined=refined_masks_exist[i_mask],
                                                                 output_path=output_sub_path)

                # save precomputed
                dat = {'syn_noises': syn_noises, 'base_color': base_color}
                save_pickle(dat, precomputed)

                # save model
                save_pickle(noise_models, model_name)

            # syn_noises are unmasked and base_color is a scalar or a 3d vector
            multilevel_syn_noises.append(syn_noises)
            base_colors.append(base_color)
        else:
            multilevel_syn_noises.append(None)
            base_colors.append(None)

    return multilevel_syn_noises, base_colors