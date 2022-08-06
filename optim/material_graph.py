import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import matplotlib.pyplot as plt

from optim.render import TextureOps
from config import default_device


class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()

    def regularize(self):
        pass

    def bypass(self):
        return self.forward()

    def scale_params(self, scale):
        pass


class NoiseGenerators(Node):
    def noise_map_to_tensor(self, noise_maps):
        noise_maps_tensor = []
        for noise_map in noise_maps:
            noise_map_ = TextureOps.numpy2tensor(noise_map)
            if self.rescale is not None:
                noise_map_ = F.interpolate(noise_map_, scale_factor=self.rescale, mode='bilinear')
            noise_maps_tensor.append(noise_map_)
        return noise_maps_tensor

    def value_to_tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def __init__(self, noise_maps, base_value, rescale, device):
        super(NoiseGenerators, self).__init__()
        self.rescale = rescale
        self.device = device
        self.noise_maps = self.noise_map_to_tensor(noise_maps)
        self.base_value = self.value_to_tensor(base_value)
        self.dim = base_value.size

    def num_of_noise_maps(self):
        return len(self.noise_maps)

    def forward(self):
        return self.noise_maps, self.base_value


class NoiseFilter(Node):
    def initialize_params(self, num, scale):
        init = torch.ones((num,), dtype=torch.float32, device=self.device) * scale
        return nn.Parameter(init)

    def __init__(self, noise_generators, device, trainable_bias=True, init_intensity=1.0, init_sigma=0.01, init_bias=0.0,
                 init_value_intensity=1.0, init_value_bias=0.0):
        super(NoiseFilter, self).__init__()
        self.device = device
        self.noise_generators = noise_generators
        self.kernel_size = (15, 15)

        # initialize parameters for noise maps
        self.intensities = self.initialize_params(num=noise_generators.num_of_noise_maps(), scale=init_intensity)
        self.sigmas = self.initialize_params(num=noise_generators.num_of_noise_maps(), scale=init_sigma)
        self.biases = self.initialize_params(num=noise_generators.num_of_noise_maps(), scale=init_bias)

        # initialize bias for base value
        self.value_intensity = self.initialize_params(num=1, scale=init_value_intensity)
        self.value_bias = self.initialize_params(num=noise_generators.dim, scale=init_value_bias)

        # do not train bias values
        if not trainable_bias:
            self.displacements.requires_grad = False
            self.value_bias.requires_grad = False

    def regularize(self):
        # self.intensities.clamp_(0.0)
        self.sigmas.clamp_(1e-3)
        # self.biases.clamp_(-0.1, 0.1)
        # self.value_bias.clamp_(-0.1, 0.1)

    def forward(self):
        noise_maps, base_value = self.noise_generators()
        outputs = []
        for noise_map, intensity, sigma, bias in zip(noise_maps, self.intensities, self.sigmas, self.biases):
            output = kornia.filters.gaussian_blur2d(noise_map * intensity + bias, kernel_size=self.kernel_size, sigma=(sigma, sigma))
            outputs.append(output)
        out = sum(outputs) + (base_value*self.value_intensity + self.value_bias).view(1, -1, 1, 1)
        return out, None

    def bypass(self):
        noise_maps, base_value = self.noise_generators()
        out = sum(noise_maps) + base_value.view(1, -1, 1, 1)
        return out, None

    # rescale parameters
    def scale_params(self, scale):
        self.sigmas.copy_(self.sigmas * scale)
        self.kernel_size = (int(self.kernel_size[0] * scale) + 1, int(self.kernel_size[1] * scale) + 1)


class MasksNode(Node):
    def mask_map_to_tensor(self, mask_maps):
        n_mask = mask_maps.shape[0]
        mask_map_tensors = []
        for i in range(n_mask):
            mask_map_tensor = TextureOps.numpy2tensor(mask_maps[i])
            if self.rescale is not None:
                mask_map_tensor = F.interpolate(mask_map_tensor, scale_factor=self.rescale, mode='bilinear')
            mask_map_tensors.append(mask_map_tensor)

        return mask_map_tensors

    def initialize_params(self, num, scale):
        init = torch.ones((num,), device=self.device, dtype=torch.float32) * scale
        return nn.Parameter(init)

    def __init__(self, mask_maps, rescale, device, init_sigma=1.0):
        super(MasksNode, self).__init__()

        self.rescale = rescale
        self.device = device

        self.mask_maps = self.mask_map_to_tensor(mask_maps)
        self.sigmas = self.initialize_params(num=len(self.mask_maps), scale=init_sigma)
        self.kernel_size = (15, 15)

    def regularize(self):
        self.sigmas.clamp_(1e-3)

    def forward(self):
        filtered = []
        for mask_map, sigma in zip(self.mask_maps, self.sigmas):
            mask = kornia.filters.gaussian_blur2d(mask_map, kernel_size=self.kernel_size, sigma=(sigma, sigma))
            filtered.append(mask)

        return filtered

    def bypass(self):
        return self.mask_maps

    def scale_params(self, scale):
        self.sigmas.copy_(self.sigmas * scale)
        self.kernel_size = (int(self.kernel_size[0] * scale) + 1, int(self.kernel_size[1] * scale) + 1)


class LinearCombiner(Node):
    def __init__(self, noise_filters, mask_node, device, auto_infer=False):
        super(LinearCombiner, self).__init__()
        self.device = device
        self.auto_infer = auto_infer
        self.noise_filters = nn.ModuleList(noise_filters)
        self.mask_node = mask_node

    def regularize(self):
        for noise_filter in self.noise_filters:
            noise_filter.regularize()
        self.mask_node.regularize()

    def infer(self, noise_maps, aggregated_mask_maps):
        if aggregated_mask_maps[0] is None:
            out = noise_maps[0] * (1 - aggregated_mask_maps[1]) + noise_maps[1]
            mask = torch.ones_like(aggregated_mask_maps[1], dtype=aggregated_mask_maps[1].dtype, device=self.device)
        else:
            out = noise_maps[0] + noise_maps[1] * (1 - aggregated_mask_maps[0])
            mask = torch.ones_like(aggregated_mask_maps[0], dtype=aggregated_mask_maps[0].dtype, device=self.device)

        return out, mask

    @staticmethod
    def check_inferable(mask_maps):
        if len(mask_maps) == 2 and ((mask_maps[0] is None and mask_maps[1] is not None) or (mask_maps[0] is not None and mask_maps[1] is None)):
            return True
        else:
            return False

    def forward(self):
        mask_maps = self.mask_node()
        noise_maps, aggregated_mask_maps = [], []
        for noise_filter in self.noise_filters:
            noise_map, aggregated_mask_map = noise_filter()
            noise_maps.append(noise_map)
            aggregated_mask_maps.append(aggregated_mask_map)

        if self.auto_infer and self.check_inferable(aggregated_mask_maps):
            return self.infer(noise_maps, aggregated_mask_maps)

        combined = []
        combined_mask = []
        for noise_filter, mask_map in zip(self.noise_filters, mask_maps):
            noise_map, aggregated_mask_map = noise_filter()
            if aggregated_mask_map is None:
                combined.append(noise_map * mask_map)
                combined_mask.append(mask_map)
            else:
                combined.append(noise_map)
                combined_mask.append(aggregated_mask_map)

        out = torch.sum(torch.cat(combined, dim=0), dim=0, keepdim=True)
        mask = torch.sum(torch.cat(combined_mask, dim=0), dim=0, keepdim=True)
        return out, mask

    def bypass(self):
        combined = []
        combined_mask = []
        mask_maps = self.mask_node.bypass()
        for noise_filter, mask_map in zip(self.noise_filters, mask_maps):
            noise_map, aggregated_mask_map = noise_filter.bypass()
            if aggregated_mask_map is None:
                combined.append(noise_map * mask_map)
                combined_mask.append(mask_map)
            else:
                combined.append(noise_map)
                combined_mask.append(aggregated_mask_map)

        out = torch.sum(torch.cat(combined, dim=0), dim=0, keepdim=True)
        mask = torch.sum(torch.cat(combined_mask, dim=0), dim=0, keepdim=True)

        return out, mask

    def scale_params(self, scale):
        for noise_filter in self.noise_filters:
            noise_filter.scale_params(scale)
        self.mask_node.scale_params(scale)


class MaterialGraph(nn.Module):
    def build_computational_graph(self, components):
        if components.is_leaf:
            albedo_noise_generators = NoiseGenerators(components.albedo_maps, components.base_albedo, self.rescale, self.device)
            height_noise_generators = NoiseGenerators(components.height_maps, components.base_height, self.rescale, self.device)
            roughness_noise_generators = NoiseGenerators(components.roughness_maps, components.base_roughness, self.rescale, self.device)
            albedo = NoiseFilter(albedo_noise_generators, self.device, init_sigma=1)
            height = NoiseFilter(height_noise_generators, self.device, init_sigma=1)
            roughness = NoiseFilter(roughness_noise_generators, self.device, init_sigma=1)
            return albedo, height, roughness

        else:
            mask_maps = components.procedural_mask_maps if self.use_procedural_mask else components.mask_maps
            albedo_mask = MasksNode(mask_maps, self.rescale, self.device, init_sigma=1)
            height_mask = MasksNode(mask_maps, self.rescale, self.device, init_sigma=1)
            roughness_mask = MasksNode(mask_maps, self.rescale, self.device, init_sigma=1)

            albedo_inputs = []
            height_inputs = []
            roughness_inputs = []
            for i, next_layer in enumerate(components.next):
                albedo_input, height_input, roughness_input = self.build_computational_graph(next_layer)
                albedo_inputs.append(albedo_input)
                height_inputs.append(height_input)
                roughness_inputs.append(roughness_input)
            albedo = LinearCombiner(albedo_inputs, albedo_mask, self.device, auto_infer=self.cfg.auto_infer)
            height = LinearCombiner(height_inputs, height_mask, self.device, auto_infer=self.cfg.auto_infer)
            roughness = LinearCombiner(roughness_inputs, roughness_mask, self.device, auto_infer=self.cfg.auto_infer)

        return albedo, height, roughness

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def __init__(self, procedural_components, cfg, rescale=None, use_procedural_mask=True, device=default_device):
        super(MaterialGraph, self).__init__()

        self.cfg = cfg
        self.rescale = rescale
        self.use_procedural_mask = use_procedural_mask
        self.device = device

        self.albedo, self.height, self.roughness = self.build_computational_graph(procedural_components)

    def forward(self):
        albedo, _ = self.albedo()
        height, _ = self.height()
        roughness, _ = self.roughness()

        return albedo, height, roughness

    def regularize(self):
        self.albedo.regularize()
        self.height.regularize()
        self.roughness.regularize()

    def print_parameters(self):
        def print_svbrdf_parameters(svbrdf, name):
            print('*' * 50)
            print(f'{name}: ')
            for name, param in svbrdf.named_parameters():
                print(f'Parameter name: {name}, val: {param.data}')

        print_svbrdf_parameters(self.albedo, 'Albedo')
        print_svbrdf_parameters(self.height, 'Height')
        print_svbrdf_parameters(self.roughness, 'Roughness')

    # bypass filter nodes
    def naive(self):
        albedo, _ = self.albedo.bypass()
        height, _ = self.height.bypass()
        roughness, _ = self.roughness.bypass()

        return albedo, height, roughness

    def scale_params(self, scale):
        self.albedo.scale_params(scale)
        self.height.scale_params(scale)
        self.roughness.scale_params(scale)
