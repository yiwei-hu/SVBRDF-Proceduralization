from abc import ABC, abstractmethod
import numpy as np
from numpy.fft import fftshift, ifftshift
from noise.noise import match_histograms, random_phase_noise_mosaic

upscaling_psd = False


def psd_upscaling(psd, scale):
    if scale <= 1:
        return psd
    assert (psd.shape[0] == psd.shape[1])
    size = psd.shape[0]
    target_size = int(size*scale)
    psd_ = fftshift(psd)
    pad_width = (target_size - size) // 2
    psd_padded = np.pad(psd_, (pad_width, pad_width))
    upscaled_psd = ifftshift(psd_padded)

    return upscaled_psd


class Node(ABC):
    @abstractmethod
    def __call__(self, size):
        pass


class ProceduralMaps(Node):
    def __init__(self, noise_models, base_color):
        self.noise_models = noise_models
        self.base_color = base_color

    def __call__(self, size):
        if isinstance(size, int):
            size = (size, size)
        syn_noises = []
        for noise_model in self.noise_models:
            noise_map = noise_model(size)
            syn_noises.append(noise_map)
        return syn_noises, self.base_color


class HistogramMatching(Node):
    def __init__(self, noise_model, target_image, target_mask, multichannel):
        self.noise_model = noise_model
        self.target_image = target_image
        self.target_mask = target_mask
        self.multichannel = multichannel

    def __call__(self, size):
        if isinstance(size, int):
            size = (size, size)
        out = self.noise_model(size)
        if self.target_image is not None:
            placeholder = np.ones(size, dtype=np.bool)
            out = match_histograms(out, self.target_image, placeholder, self.target_mask, self.multichannel)
        return out


class Noise(Node):
    def __init__(self, psd, normalizer=None, alpha=0.05, op=np.abs):
        self.psd = psd
        self.normalizer = normalizer
        self.alpha = alpha
        self.op = op

    def __call__(self, size):
        if isinstance(size, int):
            size = (size, size)

        # compatibility purpose
        try:
            self.alpha
        except:
            self.alpha = 0.05
        try:
            self.op
        except:
            self.op = np.abs

        if upscaling_psd:
            psd = psd_upscaling(self.psd, size[0] / 512)
        else:
            psd = self.psd

        out = random_phase_noise_mosaic(psd, size, alpha=0.5, op=self.op)
        if self.normalizer is not None:
            out = self.normalizer.denormalize(out)

        return out


class NoiseColored(Node):
    def __init__(self, psds, pca_model, normalizer, prenormalizer=None, alpha=0.05, op=np.abs):
        self.psds = psds
        self.nc = len(psds)
        self.pca_model = pca_model
        self.normalizer = normalizer
        self.prenormalizer = prenormalizer
        self.alpha = alpha
        self.op = op

    def __call__(self, size):
        if isinstance(size, int):
            size = (size, size)

        # compatibility purpose
        try:
            self.alpha
        except:
            self.alpha = 0.05
        try:
            self.op
        except:
            self.op = np.abs

        # hard code here
        if upscaling_psd:
            psds = []
            for psd in self.psds:
                psds.append(psd_upscaling(psd, size[0] / 512))
        else:
            psds = self.psds

        out_pca = []
        for psd in psds:
            out_pca.append(random_phase_noise_mosaic(psd, size, alpha=self.alpha, op=self.op))

        out_pca = np.stack(out_pca, axis=2)

        out_pca = self.normalizer.denormalize(out_pca)
        out = self.pca_model.inverse_transform(out_pca.reshape((-1, self.nc))).reshape((*size, self.nc))
        if self.prenormalizer is not None:
            out = self.prenormalizer.denormalize(out)
        return out

