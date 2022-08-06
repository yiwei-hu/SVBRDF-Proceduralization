import copy
import numpy as np
import torch
import torch.nn as nn
import kornia
import torchvision.models as models
from collections import Counter
from config import default_device
from optim.render import Height2Normal, Renderer


class ReconstructionLoss:
    def __init__(self, albedo, height, roughness, intensity, cfg):
        self.albedo = albedo
        self.height = height
        self.height3 = height.expand_as(albedo)
        self.roughness = roughness
        self.roughness3 = roughness.expand_as(albedo)
        self.intensity = intensity
        self.cfg = cfg

        self.height2normal = Height2Normal(intensity)
        self.normal = self.height2normal(self.height) * 0.5 + 0.5

        self.res = self.albedo.shape[2]
        self.renderer = Renderer(self.res, normal_intensity=intensity)
        with torch.no_grad():
            self.render_targets = self.renderer.render(self.albedo, self.height, self.roughness, return_index=False)

        self.criterion_pixel = nn.L1Loss() if cfg.use_l1 else nn.MSELoss()
        self.criterion_SSIM = SSIMLoss()
        if cfg.match_height:
            self.real_svbrdf = torch.cat((self.albedo, self.normal, self.roughness3, self.height3), dim=0)
            self.n_svbrdf = 4
            all_targets = (self.albedo, self.normal, self.roughness3, self.height3, *self.render_targets)
        else:
            self.real_svbrdf = torch.cat((self.albedo, self.normal, self.roughness3), dim=0)
            self.n_svbrdf = 3
            all_targets = (self.albedo, self.normal, self.roughness3, *self.render_targets)

        self.criterion_vgg = PrecomputedVGGLoss(all_targets, cfg.feature_layers, cfg.feature_weights,
                                                cfg.style_layers, cfg.style_weights, cfg.use_l1)

    def compute_loss(self, x, y, y_index):
        pixel_loss = self.criterion_pixel(x, y)
        ssim_loss = self.criterion_SSIM(x, y)
        vgg_loss = self.criterion_vgg.eval(x, y_index)
        loss = pixel_loss * self.cfg.pixel_loss_weight + ssim_loss * self.cfg.ssim_loss_weight + vgg_loss * self.cfg.vgg_loss_weight

        return loss, {'tot': loss.item(), 'pixel': pixel_loss.item(), 'ssim': ssim_loss.item(), 'vgg': vgg_loss.item()}

    def eval(self, material, n_render):
        albedo, height, roughness = material
        normal = self.height2normal(height) * 0.5 + 0.5
        if self.cfg.match_height:
            fake_svbrdf = torch.cat((albedo, normal, roughness.expand_as(albedo), height.expand_as(albedo)), dim=0)
        else:
            fake_svbrdf = torch.cat((albedo, normal, roughness.expand_as(albedo)), dim=0)

        svbrdf_loss, svbrdf_losses = self.compute_loss(fake_svbrdf, self.real_svbrdf, np.arange(self.n_svbrdf))

        if n_render > 0:
            render_targets, index_of_lighting = self.renderer.render(albedo, height, roughness, samples=n_render,  return_index=True)
            fake_render = torch.cat(render_targets, dim=0)
            real_render = torch.cat([self.render_targets[i] for i in index_of_lighting], dim=0)
            render_loss, render_losses = self.compute_loss(fake_render, real_render, [i + self.n_svbrdf for i in index_of_lighting])
        else:
            render_loss, render_losses = 0, {}

        loss = svbrdf_loss + render_loss*self.cfg.render_loss_weight
        losses = Counter(svbrdf_losses) + Counter(render_losses)
        return loss, losses


class Normalization(nn.Module):
    def __init__(self, mean, std, device=default_device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        self.std = torch.tensor(std, device=device).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = None

    def forward(self, feat):
        self.feature = feat
        return feat


class FeatureLoss(nn.Module):
    def __init__(self, use_l1):
        super(FeatureLoss, self).__init__()
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, x, y):
        return self.criterion(x, y)


class StyleLoss(nn.Module):
    def __init__(self, use_l1=False):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, fake, real):
        gram_fake = self.gram_matrix(fake)
        gram_real = self.gram_matrix(real)
        return self.criterion(gram_fake, gram_real)

    @staticmethod
    def gram_matrix(x):
        b, c, h, w = x.shape
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class VGGLoss(nn.Module):
    def __init__(self, feature_layers, feature_weights, style_layers, style_weights, use_l1, device=default_device):
        super(VGGLoss, self).__init__()
        self.device = device

        assert(len(feature_layers) == len(feature_weights))
        assert(len(style_layers) == len(style_weights))

        vgg = models.vgg19(pretrained=True)
        cnn = vgg.features.to(self.device).eval().requires_grad_(False)

        self.model1, self.feature_layers1, self.style_layers1 = self.get_feature_extractors(cnn, feature_layers, style_layers)
        self.model2, self.feature_layers2, self.style_layers2 = self.get_feature_extractors(cnn, feature_layers, style_layers)

        assert len(self.feature_layers1 == feature_weights)
        assert len(self.style_layers1 == style_weights)

        self.criterionFeature = FeatureLoss(use_l1)
        self.criterionStyle = StyleLoss(use_l1)
        self.feature_weights = feature_weights
        self.style_weights = style_weights

    def forward(self, fake, real):
        # run VGG model
        self.model1(fake)
        self.model2(real)

        # extract features and compute losses
        feature_loss = torch.tensor([0.0], requires_grad=True, device=self.device)
        for extractor1, extractor2, w in zip(self.feature_layers1, self.feature_layers2, self.feature_weights):
            feature_loss += self.criterionFeature(extractor1.feature, extractor2.feature.detach()) * w

        style_loss = torch.tensor([0.0], requires_grad=True, device=self.device)
        for extractor1, extractor2, w in zip(self.style_layers1, self.style_layers2, self.style_weights):
            style_loss += self.criterionStyle(extractor1.feature, extractor2.feature.detach()) * w

        return feature_loss, style_loss

    @staticmethod
    def get_feature_extractors(cnn, feature_layer_names, style_layer_names, device=default_device):
        cnn = copy.deepcopy(cnn)

        # normalization module for ImageNet
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device)
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

        # just in order to have an iterable access to or list of content/style
        # losses
        feature_layers = []
        style_layers = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        for number, layer in cnn.named_children():
            if isinstance(layer, nn.Conv2d):
                name = VGG19_layer_mapping[number]
                assert('conv' in name)
            elif isinstance(layer, nn.ReLU):
                name = VGG19_layer_mapping[number]
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
                assert ('relu' in name)
            elif isinstance(layer, nn.MaxPool2d):
                name = VGG19_layer_mapping[number]
                assert ('pool' in name)
            elif isinstance(layer, nn.BatchNorm2d):
                name = VGG19_layer_mapping[number]
                assert('bn' in name)
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in feature_layer_names:
                feature_extractor = FeatureExtractor()
                model.add_module(f'feature_layers_{feature_layer_names.index(name)}', feature_extractor)
                feature_layers.append(feature_extractor)

            if name in style_layer_names:
                feature_extractor = FeatureExtractor()
                model.add_module("style_layers_{}".format(style_layer_names.index(name)), feature_extractor)
                style_layers.append(feature_extractor)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], FeatureExtractor):
                break

        model = model[:(i + 1)]

        return model, feature_layers, style_layers


class PrecomputedVGGLoss:
    def __init__(self, real_targets, feature_layers, feature_weights, style_layers, style_weights, use_l1, device=default_device):
        self.device = device

        self.n_feat = len(feature_layers)
        self.n_style = len(style_layers)

        vgg = models.vgg19(pretrained=True)
        cnn = vgg.features.to(self.device).eval().requires_grad_(False)

        self.model, self.feature_layers, self.style_layers = VGGLoss.get_feature_extractors(cnn, feature_layers, style_layers)

        assert len(self.feature_layers) == self.n_feat
        assert len(self.style_layers) == self.n_style

        real_targets = torch.cat(real_targets, dim=0)
        with torch.no_grad():
            self.target_features, self.target_styles = self.precompute(real_targets)

        self.criterionFeature = FeatureLoss(use_l1)
        self.criterionStyle = StyleLoss(use_l1)
        self.feature_weights = feature_weights
        self.style_weights = style_weights

    def precompute(self, real):
        self.model(real)
        features = []
        for extractor in self.feature_layers:
            features.append(extractor.feature.detach())

        styles = []
        for extractor in self.style_layers:
            styles.append(extractor.feature.detach())
        return features, styles

    def eval(self, x, idx):
        # run VGG model
        self.model(x)

        # extract features and compute losses
        feature_loss = torch.tensor([0.0], device=self.device)
        for extractor, all_features, w in zip(self.feature_layers, self.target_features, self.feature_weights):
            feature = all_features[idx, :, :, :]
            feature_loss += self.criterionFeature(extractor.feature, feature) * w

        style_loss = torch.tensor([0.0], device=self.device)
        for extractor, all_features, w in zip(self.style_layers, self.target_styles, self.style_weights):
            feature = all_features[idx, :, :, :]
            # print(self.criterionStyle(extractor.feature, feature), w)
            style_loss += self.criterionStyle(extractor.feature, feature) * w

        return feature_loss + style_loss


class SSIMLoss(nn.Module):
    def __init__(self,  window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        return kornia.losses.ssim_loss(x, y, self.window_size)


VGG19_layer_mapping = {'0': 'conv1_1',
                       '1': 'relu1_1',
                       '2': 'conv1_2',
                       '3': 'relu1_2',
                       '4': 'pool1',
                       '5': 'conv2_1',
                       '6': 'relu2_1',
                       '7': 'conv2_2',
                       '8': 'relu2_2',
                       '9': 'pool2',
                       '10': 'conv3_1',
                       '11': 'relu3_1',
                       '12': 'conv3_2',
                       '13': 'relu3_2',
                       '14': 'conv3_3',
                       '15': 'relu3_3',
                       '16': 'conv3_4',
                       '17': 'relu3_4',
                       '18': 'pool3',
                       '19': 'conv4_1',
                       '20': 'relu4_1',
                       '21': 'conv4_2',
                       '22': 'relu4_2',
                       '23': 'conv4_3',
                       '24': 'relu4_3',
                       '25': 'conv4_4',
                       '26': 'relu4_4',
                       '27': 'pool4',
                       '28': 'conv5_1',
                       '29': 'relu5_1',
                       '30': 'conv5_2',
                       '31': 'relu5_2',
                       '32': 'conv5_3',
                       '33': 'relu5_3',
                       '34': 'conv5_4',
                       '35': 'relu5_4',
                       '36': 'pool5'}


VGG19_layer_mapping_inverse = {'conv1_1': '0',
                               'relu1_1': '1',
                               'conv1_2': '2',
                               'relu1_2': '3',
                               'pool1': '4',
                               'conv2_1': '5',
                               'relu2_1': '6',
                               'conv2_2': '7',
                               'relu2_2': '8',
                               'pool2': '9',
                               'conv3_1': '10',
                               'relu3_1': '11',
                               'conv3_2': '12',
                               'relu3_2': '13',
                               'conv3_3': '14',
                               'relu3_3': '15',
                               'conv3_4': '16',
                               'relu3_4': '17',
                               'pool3': '18',
                               'conv4_1': '19',
                               'relu4_1': '20',
                               'conv4_2': '21',
                               'relu4_2': '22',
                               'conv4_3': '23',
                               'relu4_3': '24',
                               'conv4_4': '25',
                               'relu4_4': '26',
                               'pool4': '27',
                               'conv5_1': '28',
                               'relu5_1': '29',
                               'conv5_2': '30',
                               'relu5_2': '31',
                               'conv5_3': '32',
                               'relu5_3': '33',
                               'conv5_4': '34',
                               'relu5_4': '35',
                               'pool5': '36'}