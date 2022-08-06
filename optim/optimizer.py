import os
import os.path as pth
import numpy as np
import torch
import torch.optim as optim
import cv2
import visdom

from optim.material_graph import MaterialGraph
from optim.loss import ReconstructionLoss
from utils import write_image, make_grid, record_losses
from optim.render import Renderer, TextureOps, height2normal


class MaterialGraphOptimizer:
    @staticmethod
    def init_real_data(image, target_size=None):
        if target_size is not None:
            image = cv2.resize(image, target_size)
        return TextureOps.numpy2tensor(image)

    def __init__(self, generators, target_images, normal_intensity, cfg, save_path, use_procedural_mask=True):
        self.generators = generators
        self.target_images = target_images
        self.normal_intensity = normal_intensity
        self.cfg = cfg
        self.save_path = save_path

        rescale = None if cfg.target_size is None else cfg.target_size[0] / 512
        self.material = MaterialGraph(generators, cfg, rescale, use_procedural_mask)

        self.albedo = self.init_real_data(target_images['albedo'], cfg.target_size)
        self.height = self.init_real_data(target_images['height'], cfg.target_size)[:, 0:1, :, :]
        self.roughness = self.init_real_data(target_images['roughness'], cfg.target_size)[:, 0:1, :, :]

        self.loss_obj = ReconstructionLoss(self.albedo, self.height, self.roughness, self.normal_intensity, cfg)

        self.res = self.albedo.shape[2]
        self.renderer = Renderer(self.res, normal_intensity=normal_intensity)

    def optimize(self):
        vis = visdom.Visdom(port=self.cfg.display_port)
        loss_graph = {}

        output_path = pth.join(self.save_path, 'optim')
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            self.save_material_maps_as_grid(self.material.naive(), n_samples=(0, 8, 12), output_path=output_path, suffix='_linear')
            self.save_material_maps_as_grid(self.material(), n_samples=(0, 8, 12), output_path=output_path, suffix='_init')

        print("Before optimization:")
        self.material.print_parameters()

        optimizer = optim.Adam(self.material.parameters(), lr=self.cfg.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.milestones, gamma=0.25)

        print('Optimizing...')
        for i in range(self.cfg.n_iter):
            optimizer.zero_grad()

            material = self.material()
            loss, losses = self.loss_obj.eval(material, self.cfg.n_renders)
            record_losses(loss_graph, losses)

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.material.regularize()

            if i % self.cfg.vis_every == 0 or i == self.cfg.n_iter - 1:
                print(f'[{i}/{self.cfg.n_iter}], loss = {loss.item()}, pixel: {losses["pixel"]}, '
                      f'SSIM: {losses["ssim"]}, vgg: {losses["vgg"]}, lr: {scheduler.get_last_lr()}')

                for loss_name, loss_list in loss_graph.items():
                    vis.line(np.asarray(loss_list), X=np.arange(len(loss_list)), win=loss_name, opts=dict(title=loss_name))

                with torch.no_grad():
                    material_maps = self.material()
                    self.save_material_maps_as_grid(material_maps, n_samples=(0, 8, 12), output_path=output_path, suffix=f'{i:05d}')

        print("After optimization:")
        self.material.print_parameters()
        self.material.save(pth.join(self.save_path, 'graph.torch'))

    def visualize(self, output_path=None):
        if output_path is None:
            output_path = self.save_path

        with torch.no_grad():
            material = self.material()
            loss, losses = self.loss_obj.eval(material, n_render=0)
            print(f'Optimized: Loss = {loss.item()}, pixel: {losses["pixel"]}, SSIM: {losses["ssim"]}, vgg: {losses["vgg"]}')
            material_linear = self.material.naive()
            loss, losses = self.loss_obj.eval(material_linear, n_render=0)
            print(f'Linear: Loss = {loss.item()}, pixel: {losses["pixel"]}, SSIM: {losses["ssim"]}, vgg: {losses["vgg"]}')

            self.save_material_maps((self.albedo, self.height, self.roughness), self.normal_intensity, output_path, suffix='(gt)')
            self.save_material_maps(material, self.normal_intensity,output_path)
            self.save_material_maps(material_linear, self.normal_intensity, output_path, suffix='(raw)')

            grid_target = self.save_material_maps_as_grid((self.albedo, self.height, self.roughness), n_samples=(0, 8, 12), output_path=None)
            grid = self.save_material_maps_as_grid(material, n_samples=(0, 8, 12), output_path=None)
            grid_linear = self.save_material_maps_as_grid(material_linear, n_samples=(0, 8, 12), output_path=None)

            gap = np.ones((5, *grid.shape[1:]))
            grid_out = np.concatenate((grid_target, gap, grid, gap, grid_linear), axis=0)
            write_image(np.clip(grid_out, 0, 1), pth.join(output_path, f'comparison.png'))

    def save_material_maps_as_grid(self, material_maps, n_samples, output_path, suffix=''):
        albedo, height, roughness = material_maps
        if isinstance(n_samples, int):
            samples = np.arange(n_samples)
        else:
            samples = n_samples
        render_targets = self.renderer.render(albedo, height, roughness, samples=samples, return_index=False)
        albedo = TextureOps.tensor2numpy(albedo)
        height = TextureOps.tensor2numpy(height).squeeze()
        roughness = TextureOps.tensor2numpy(roughness).squeeze()
        normal = height2normal(height, intensity=self.normal_intensity)
        grid = make_grid((albedo, normal, height, roughness, *[TextureOps.tensor2numpy(render_target) for render_target in render_targets]), n_row=10)
        if output_path is not None:
            write_image(np.clip(grid, 0, 1), pth.join(output_path, f'material{suffix}.png'))
        else:
            return grid

    @staticmethod
    def save_material_maps(material_maps, normal_intensity, output_path, suffix=''):
        albedo, height, roughness = material_maps
        albedo = TextureOps.tensor2numpy(albedo)
        height = TextureOps.tensor2numpy(height).squeeze()
        roughness = TextureOps.tensor2numpy(roughness).squeeze()

        write_image(np.clip(albedo, 0, 1), pth.join(output_path, f'albedo{suffix}.png'))
        np.save(pth.join(output_path, f'height{suffix}.npy'), height)
        # clipped height map, should never use this, only for visualization
        write_image(np.clip(height, 0, 1), pth.join(output_path, f'height{suffix}.png'))
        normal = height2normal(height, intensity=normal_intensity)
        write_image(normal, pth.join(output_path, f'normal{suffix}.png'))
        write_image(np.clip(roughness, 0, 1), pth.join(output_path, f'roughness{suffix}.png'))

