import os
import random
import torch
import torch.nn as nn
import numpy as np
import kornia
import cv2
from config import default_device


def loadLightAndCamera(in_dir):
    print('Load camera position from ', os.path.join(in_dir, 'camera_pos.txt'))
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    print('Load light position from ', os.path.join(in_dir, 'light_pos.txt'))
    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',')

    return light_pos, camera_pos, im_size, light


class Renderer:
    def print_light_config(self):
        print('Rendering Configuration: ')
        print('Number of Lights', self.n_lights)
        print('Light Position', self.lp)
        print('Camera Position:', self.cp)
        print('Image size:', self.im_size)
        print('Light power:', self.li)

    def __init__(self, res, normal_intensity, light_settings=None, device=default_device):
        # load predefined lighting configurations
        if light_settings is None:
            self.lp, self.cp, self.im_size, self.li = loadLightAndCamera('./lighting')
            self.li *= 3.1415926
        else:
            self.lp = light_settings['lp']
            self.cp = light_settings['cp']
            self.im_size = light_settings['im_size']
            self.li = light_settings['li']
            self.li *= 3.1415926

        self.lp = torch.as_tensor(self.lp, dtype=torch.float32, device=device)
        self.cp = torch.as_tensor(self.cp, dtype=torch.float32, device=device)
        self.li = torch.as_tensor(self.li, dtype=torch.float32, device=device)
        self.n_lights = self.lp.shape[0]
        # self.print_light_config()

        self.shader = Microfacet(res=res, size=self.im_size)
        self.height2normal = Height2Normal(intensity=normal_intensity)

    def render(self, albedo, height, roughness, samples=0, is_height=True, return_index=True):
        assert albedo.shape[1] == 3 and roughness.shape[1] == 1
        assert (height.shape[1] == 1 and is_height) or (height.shape[1] == 3 and not is_height)

        if isinstance(samples, int):
            if samples == 0 or samples == self.n_lights:
                index_of_lighting = np.arange(self.n_lights)
            else:
                index_of_lighting = random.sample(range(self.n_lights), min(samples, self.n_lights))
        else:
            index_of_lighting = samples

        albedo, normal, roughness = self.preprocess(albedo, height, roughness, is_height)

        render_targets = []
        for i in index_of_lighting:
            render = self.shader.eval(albedo, normal, roughness, lightPos=self.lp[i, :], cameraPos=self.cp[i, :], light=self.li)
            render_targets.append(render)

        if return_index:
            return render_targets, index_of_lighting
        else:
            return render_targets

    def preprocess(self, basecolor, height, roughness, is_height):
        albedo = basecolor.clamp(0, 1)
        normal = self.height2normal(height) if is_height else height*2.0 - 1.0
        rough = roughness.clamp(0, 1)
        return albedo, normal, rough


class Microfacet:
    def __init__(self, res, size, f0=0.04, device=default_device):
        self.res = res
        self.size = size
        self.f0 = f0
        self.eps = 1e-6
        self.device = device

        self.initGeometry()

    def initGeometry(self):
        tmp = torch.arange(self.res, dtype=torch.float32, device=self.device)
        tmp = ((tmp + 0.5) / self.res - 0.5) * self.size
        y, x = torch.meshgrid((tmp, tmp))
        self.pos = torch.stack((x, -y, torch.zeros_like(x)), 2)
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return torch.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel(self, cos, f0):
        return f0 + (1 - f0) * (1 - cos)**5

    def Fresnel_S(self, cos, specular):
        sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert(vec.size(0)==self.N)
        assert(vec.size(1)==3)
        assert(vec.size(2)==self.res)
        assert(vec.size(3)==self.res)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    def getDir(self, pos):
        vec = (pos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1,3,-1,-1)

    def AdotB(self, a, b):
        ab = (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
        return ab

    def eval(self, albedo, normal, rough, lightPos, cameraPos, light):
        self.N = albedo.shape[0]

        light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)

        v, _ = self.getDir(cameraPos)
        l, dist_l_sq = self.getDir(lightPos)
        h = self.normalize(l + v)

        n_dot_v = self.AdotB(normal, v)
        n_dot_l = self.AdotB(normal, l)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(v, h)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough**2)
        # D = self.Beckmann(n_dot_h, rough**2)

        F = self.Fresnel(v_dot_h, self.f0)
        G = self.Smith(n_dot_v, n_dot_l, rough**2)

        # lambert brdf
        f1 = albedo / np.pi

        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = f * geom * light

        return img.clamp(self.eps, 1)


class Height2Normal(nn.Module):
    def __init__(self, intensity):
        super(Height2Normal, self).__init__()
        self.intensity = intensity

    @staticmethod
    def normalize(m):
        return m / torch.sqrt(torch.sum(m ** 2, dim=1, keepdim=True))

    def forward(self, height):
        assert height.ndim == 4 and height.shape[1] == 1
        gradient = kornia.filters.spatial_gradient(height, normalized=False)
        dx = gradient[:, :, 0, :, :]
        dy = gradient[:, :, 1, :, :]

        x = -dx*self.intensity
        y = dy*self.intensity
        z = torch.ones_like(dx)
        normal = torch.cat((x, y, z), dim=1)
        normal = Height2Normal.normalize(normal)

        return normal


def height2normal(height, intensity):
    def normalize(m):
        return m / np.sqrt(np.sum(m ** 2, axis=2, keepdims=True))

    assert (height.ndim == 2)

    dx = cv2.Sobel(height, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(height, cv2.CV_64F, 0, 1, ksize=3)

    normal = np.zeros((*height.shape, 3, ), dtype=np.float32)
    normal[:, :, 0] = -dx * intensity
    normal[:, :, 1] = dy * intensity
    normal[:, :, 2] = 1
    normal = normalize(normal)
    normal = normal * 0.5 + 0.5

    return normal


class TextureOps:
    @staticmethod
    def tensor2numpy(x):
        assert x.ndim == 4
        return x.detach().squeeze(0).permute((1, 2, 0)).cpu().numpy()

    @staticmethod
    def numpy2tensor(x, device=default_device):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)
        x = torch.as_tensor(x, dtype=torch.float32, device=device).permute((2, 0, 1)).unsqueeze(0)
        return x



