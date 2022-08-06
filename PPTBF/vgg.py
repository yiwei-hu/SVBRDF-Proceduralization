import math
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

from PPTBF.descriptors import LocalBinaryPattern


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features.eval()
        # print(features)
        self.slice1 = nn.Sequential(Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # repeat the color channel 3 times
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # feed the input image to VGG16
        output = self.slice1(x)
        relu1_2 = output
        output = self.slice2(output)
        relu2_2 = output
        output = self.slice3(output)
        relu3_3 = output
        output = self.slice4(output)
        relu4_3 = output
        output_features = namedtuple('features', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])

        return output_features(relu1_2, relu2_2, relu3_3, relu4_3)


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features.eval()
        # print(features)
        self.slice1 = nn.Sequential(Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # repeat the color channel 3 times
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # feed the input image to VGG19
        output = self.slice1(x)
        relu1_2 = output
        output = self.slice2(output)
        relu2_2 = output
        output = self.slice3(output)
        relu3_4 = output

        output_features = namedtuple('features', ['relu1_2', 'relu2_2', 'relu3_4'])

        return output_features(relu1_2, relu2_2, relu3_4)


def image_loader(image_name, crop_size=0, interpolation=Image.BILINEAR):
    if crop_size == 0:
        loader = transforms.Compose([
            transforms.Resize((224, 224), interpolation=interpolation), # scale input images to fit the VGG input
            transforms.ToTensor()
        ])
    else:
        loader = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Resize((224, 224), interpolation=interpolation), # scale input images to fit the VGG input
            transforms.ToTensor()
        ])

    # load greyscale images
    image = Image.open(image_name).convert('L')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    # print(image.shape)
    return image.to(torch.float)


def from_numpy(image_np, crop_size=0, interpolation=Image.BILINEAR):
    if crop_size == 0:
        loader = transforms.Compose([
            transforms.Resize((224, 224), interpolation=interpolation), # scale input images to fit the VGG input
            transforms.ToTensor()
        ])
    else:
        loader = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Resize((224, 224), interpolation=interpolation), # scale input images to fit the VGG input
            transforms.ToTensor()
        ])
    # convert to PIL image
    image = Image.fromarray((np.clip(image_np, 0, 1)*255.0).astype(np.uint8))
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    # print(image.shape)
    return image.to(torch.float)


def gram_matrix(x):
    # b = batch size = 1, ch = number of feature maps, h = height, w = width
    (b, ch, h, w) = x.shape
    features = x.view(b * ch, h * w)
    # compute the gram product
    G = torch.mm(features, features.t())
    # normalize the values of the gram matrix
    return G.div(b * ch * h * w)


def style_feature(feature):
    style_weights = [math.sqrt(1e6 / n ** 2) for n in [64, 128, 256]]
    features = [torch.flatten(style_weights[i] * gram_matrix(feature[i]).detach()) for i in range(len(feature))]
    return reduce(lambda x, y: torch.cat([x, y]), features)


def fft_feature(image):
    if not torch.is_tensor(image):
        raise NotImplementedError(f'unsupported input type: {type(image)}')

    image = image.squeeze()
    amplitude = torch.abs(torch.fft.fftn(image, norm='ortho'))

    return amplitude.view(-1)


def feature_vector(image, vgg, weight_lbp=1.0, weight_fft=1.0):
    lbp = LocalBinaryPattern(n_points=24, radius=3)
    style = style_feature(vgg(image)).cpu().numpy()
    fft = fft_feature(image).cpu().numpy()
    feature = np.concatenate((style, weight_lbp * lbp(image), weight_fft * fft), axis=0)

    return feature


def loss(source, target, vgg, weight_fft=1.0, weight_lbp=1.0, loss_type='l2'):
    lbp = LocalBinaryPattern(n_points=24, radius=3)
    style_src = style_feature(vgg(source))
    fft_src = fft_feature(source)

    style_dst = style_feature(vgg(target))
    fft_dst = fft_feature(target)

    p = lbp(source)
    q = lbp(target)

    if loss_type == 'l2':
        return F.mse_loss(style_src, style_dst, reduction='sum'), weight_fft * F.mse_loss(fft_src, fft_dst, reduction='sum'), weight_lbp * sum((p - q) ** 2)
    else:
        return F.l1_loss(style_src, style_dst, reduction='sum'), weight_fft * F.l1_loss(fft_src, fft_dst, reduction='sum'), weight_lbp * sum((p - q) ** 2)
