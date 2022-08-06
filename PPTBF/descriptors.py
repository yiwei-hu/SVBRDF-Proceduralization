import numpy as np
import torch
from skimage import feature


class LocalBinaryPattern:
    def __init__(self, n_points, radius):
        self.n_points = n_points
        self.radius = radius

    def __call__(self, image: torch.Tensor):
        image = image.squeeze().cpu().numpy()
        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        n_bins = 256
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
