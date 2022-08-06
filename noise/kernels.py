from numba import njit, prange
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
import cv2
from skimage.exposure import histogram_matching


# Gaussian filter with mask
def gaussian_filter_with_mask(image, mask, ksize, sigma):
    assert(ksize > 0 and ksize % 2 == 1 and sigma >= 0)

    gaussian_kernel_1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    gaussian_kernel = gaussian_kernel_1d @ gaussian_kernel_1d.transpose()

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    output = filter2d_masked(image, mask, gaussian_kernel)

    return output.squeeze()


@njit(parallel=True)
def filter2d_masked(image, mask, kernel):
    width, height, channel = image.shape
    ksize, _ = kernel.shape

    left_step = -(ksize // 2)
    right_step = ksize // 2 + 1

    output = np.zeros((width, height, channel))
    for i in prange(width):
        for j in prange(height):
            if mask[i, j]:
                val = np.zeros((channel,))
                weight = 0
                for x in prange(left_step, right_step):
                    for y in prange(left_step, right_step):
                        if 0 <= i + x < width and 0 <= j + y < height and mask[i + x, j + y]:
                            val += image[i + x, j + y] * kernel[left_step + x, left_step + y]
                            weight += kernel[left_step + x, left_step + y]
                output[i, j] = val / weight

    return output


# a FFT-based estimator
def estimate_local_psd(image, mask, T, threshold, step):
    threshold_ = int(T * T * threshold)

    mask_areas = compute_masked_pixels(mask, T)
    mask_areas_size_x, mask_areas_size_y = mask_areas.shape

    num_samples = 0
    psd = 0
    weights = 0
    for block_x in range(0, mask_areas_size_x, step):
        for block_y in range(0, mask_areas_size_y, step):
            mask_area = mask_areas[block_x, block_y]
            if mask_area >= threshold_:
                psd += np.abs(fft2(image[block_x: block_x + T, block_y: block_y + T]))*mask_area
                weights += mask_area
                num_samples += 1

    print(f'{num_samples} valid samples generated for local PSD estimation')
    return psd / weights / (T * T)


@njit(parallel=True)
def compute_valid_samples(mask, T, threshold, step):
    threshold = int(T*T*threshold)
    masked_pixels = compute_masked_pixels(mask, T)
    masked_pixel_size_x, masked_pixel_size_y = masked_pixels.shape

    num_x = (masked_pixel_size_x - 1) // step + 1
    num_y = (masked_pixel_size_y - 1) // step + 1

    n_valid_samples = 0
    for idx_x in prange(num_x):
        block_x = idx_x*step
        for idx_y in prange(num_y):
            block_y = idx_y*step
            mask_area = masked_pixels[block_x, block_y]
            if mask_area >= threshold:
                n_valid_samples += 1

    return n_valid_samples


@njit(parallel=True)
def compute_masked_pixels(mask, T):
    mask_size_x, mask_size_y = mask.shape
    masked_pixel_size_x = mask_size_x - T + 1
    masked_pixel_size_y = mask_size_y - T + 1
    masked_pixels = np.zeros((masked_pixel_size_x, masked_pixel_size_y), np.int32)

    # brute-force enumerating, unoptimized
    for block_x in prange(0, masked_pixel_size_x):
        for block_y in prange(0, masked_pixel_size_y):
            masked_pixel = 0
            for x in prange(0, T):
                for y in prange(0, T):
                    if mask[block_x + x, block_y + y] == 1:
                        masked_pixel += 1
            masked_pixels[block_x, block_y] = masked_pixel

    return masked_pixels


@njit(parallel=True)
def compute_blending_factor(t_w, t_h, blending_size):
    blending = np.zeros((t_w, t_h))
    for x in prange(t_w):
        vx = 1.0
        if x < blending_size:
            vx = (x + 1.0) / (blending_size + 1.0)
        if x >= t_w - blending_size:
            vx = (t_w - x) / (blending_size + 1.0)

        for y in prange(t_h):
            vy = 1.0
            if y < blending_size:
                vy = (y + 1.0) / (blending_size + 1.0)
            if y >= t_h - blending_size:
                vy = (t_h - y) / (blending_size + 1.0)

            blending[x, y] = vx * vy

    return blending


def masked_variance(image, mask):
    masked_image = image[mask]
    img_mean = np.average(masked_image, axis=0)
    img_var = np.sum((masked_image - img_mean)**2) / masked_image.shape[0] # masked_image.size
    return img_var


def random_phase_noise_mosaic(psd, image_size, alpha, op=np.abs):
    t_w, t_h = psd.shape
    assert(t_w == t_h)
    im_w, im_h = image_size
    result = np.zeros(image_size)

    blending_size = int(t_w * alpha)
    blending = compute_blending_factor(t_w, t_h, blending_size)

    psd_ = psd*psd.size
    for x in range(-blending_size, im_w + blending_size, t_w - blending_size):
        for y in range(-blending_size, im_h + blending_size, t_h - blending_size):
            phase = random_phase(t_w)
            phase = ifftshift(phase)

            local_noise_spectrum = psd_*np.exp(1j*phase)
            local_noise = op(ifft2(local_noise_spectrum))
            blended_noise = blending*local_noise

            i = np.arange(t_w)
            j = np.arange(t_h)
            ii, jj = np.meshgrid(i, j)
            ii = ii.flatten()
            jj = jj.flatten()
            pi, pj = x + ii, y + jj
            idx = (pi >= 0) & (pi < im_w) & (pj >= 0) & (pj < im_h)
            vpi, vpj, vii, vjj = pi[idx], pj[idx], ii[idx], jj[idx]
            result[vpi, vpj] += blended_noise[vii, vjj]

    return result


@njit
def random_phase(size):
    half = (size - 1) // 2
    full = half*2 + 1
    '''
     A  d -B
     c  0 -c 
     B -d -A
    '''
    A = (np.random.rand(half, half)*2.0 - 1.0)*np.pi
    B = (np.random.rand(half, half)*2.0 - 1.0)*np.pi
    c = (np.random.rand(half) * 2.0 - 1.0)*np.pi
    d = (np.random.rand(half) * 2.0 - 1.0) * np.pi
    phase = np.zeros((full, full))
    phase[:half, :half] = A
    phase[half + 1:, half + 1:] = -np.flip(A)
    phase[half + 1:, :half] = B
    phase[:half, half + 1:] = -np.flip(B)
    phase[half, :half] = c
    phase[half, half + 1:] = -np.flip(c)
    phase[:half, half] = d
    phase[half + 1:, half] = -np.flip(d)

    if size % 2 == 1:
        return phase
    else:
        phase_padded = np.zeros((size, size))
        phase_padded[1:, 1:] = phase
        return phase_padded


def match_histograms(I1, I2, mask1, mask2, multichannel=False):
    X1 = I1[mask1]
    X2 = I2[mask2]

    matched = histogram_matching.match_histograms(X1, X2, multichannel=multichannel)
    out = np.zeros_like(I1)
    out[mask1] = matched

    return out