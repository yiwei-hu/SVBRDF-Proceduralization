import numpy as np
from numpy.fft import ifftshift
from numpy.lib.scimath import sqrt as complex_sqrt
import matlab.engine
from utils import Timer


def gabor_approximation(image):
    assert image.ndim == 2
    psd = estimate_psd(image)
    psd = ifftshift(complex_sqrt(psd)) / image.size
    psd[0, 0] += np.average(image)
    return psd


def estimate_psd(image):
    eng = matlab.engine.start_matlab()

    eng.cd('noise/gabor')

    timer = Timer()
    timer.begin()

    psd = eng.gabor_noise(matlab.double(image.tolist()))
    psd = np.array(psd._data).reshape(psd.size, order='F')

    timer.end('PSD estimation for Gabor noise complete in')

    eng.quit()

    return psd