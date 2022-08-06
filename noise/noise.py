import os.path as pth
import math
import numpy as np
import cv2
from sklearn.decomposition import PCA
from numpy.fft import fft2
import matplotlib.pyplot as plt

from noise.kernels import masked_variance, gaussian_filter_with_mask
from noise.kernels import estimate_local_psd, compute_valid_samples, random_phase_noise_mosaic, match_histograms
from noise.gabor.gabor import gabor_approximation
from noise.procedures import Noise, NoiseColored, HistogramMatching, ProceduralMaps
from matting.patchmatch import inpaint as patchmatch_inpainting
from utils import shift_image, save_image, Normalizer, Timer


class LocalSpectrum:
    def __init__(self, image, windowsize, windowtype='p'):
        self.image = image
        self.windowsize = windowsize
        if windowtype in ['g', 'h', 'p']:
            self.windowtype = windowtype
        else:
            raise NotImplementedError(f'Unknown window type: {windowtype}')

        width, height = self.image.shape
        self.spectrum = [None]*width
        self.welch_spectrum = [None]*width
        for i in range(width):
            self.spectrum[i] = [None]*height
            self.welch_spectrum[i] = [None]*height

        self.spectrum_array = None
        self.local_guidance_map = None

    def compute_local_spectrum(self):
        timer = Timer()
        timer.begin()

        # compute local spectrum
        width, height = self.image.shape
        left_bnd = -(self.windowsize - 1) // 2
        right_bnd = self.windowsize // 2
        W = np.zeros((self.windowsize, self.windowsize))

        print(f'local spectrum window: ({left_bnd}, {right_bnd})')
        for y in range(left_bnd, right_bnd):
            for x in range(left_bnd, right_bnd):
                if self.windowtype == 'g':  # Gaussian
                    W[x - left_bnd, y - left_bnd] = LocalSpectrum.g_non_norm(x, y, self.windowsize / 3)
                elif self.windowsize == 'h':  # hamming
                    W[x - left_bnd, y - left_bnd] = LocalSpectrum.hamming(x, y, self.windowsize)
                else:  # porte
                    W[x - left_bnd, y - left_bnd] = 1

        for i in range(width):
            for j in range(height):
                ic = max(min(i, width - right_bnd), -left_bnd)
                jc = max(min(j, height - right_bnd), -left_bnd)

                assert (0 <= ic + left_bnd <= width)
                assert (0 <= ic + right_bnd <= width)
                assert (0 <= jc + left_bnd <= height)
                assert (0 <= jc + right_bnd <= height)

                sub_image = self.image[ic + left_bnd:ic + right_bnd, jc + left_bnd:jc + right_bnd].copy()
                sub_image *= W

                sub_image_fft = fft2(sub_image)
                self.spectrum[i][j] = np.abs(sub_image_fft) / sub_image.size  # normalized spectrum

        timer.end('Local spectrum computation complete in')

    def run_welch_algorithm(self, welch_windowsize, welch_step):
        timer = Timer()
        timer.begin()

        width, height = self.image.shape
        welch_left_bnd = -(welch_windowsize - 1) // 2
        welch_right_bnd = welch_windowsize // 2
        welch_x = np.arange(welch_left_bnd, welch_right_bnd, welch_step)

        print(f'Welch spectrum window: ({welch_left_bnd}, {welch_right_bnd}) with step {welch_step}')
        print(f'Totally {len(welch_x)} samples: {welch_x}')

        welch_grid_x, welch_grid_y = np.meshgrid(welch_x, welch_x)
        welch_grid_x = welch_grid_x.flatten()
        welch_grid_y = welch_grid_y.flatten()
        welch_grid_size = welch_grid_x.shape[0]

        for i in range(width):
            for j in range(height):
                welch_spectrum = np.zeros((self.windowsize, self.windowsize))
                isums = np.clip(i + welch_grid_x, a_min=0, a_max=width - 1)
                jsums = np.clip(j + welch_grid_y, a_min=0, a_max=height - 1)

                for isum, jsum in np.nditer([isums, jsums]):
                    welch_spectrum += np.square(self.spectrum[isum][jsum])  # sqr(amp)->SPD

                welch_spectrum = np.sqrt(welch_spectrum / welch_grid_size)
                self.welch_spectrum[i][j] = welch_spectrum

        self.spectrum = np.asarray(self.spectrum)
        self.welch_spectrum = np.asarray(self.welch_spectrum)
        self.spectrum_array = self.welch_spectrum.reshape((width*height, self.windowsize*self.windowsize))

        timer.end('Welch algorithm complete in')

    @staticmethod
    def g_non_norm(x, y, sigma):
        return math.exp(-(x * x + y * y) / (2 * sigma * sigma))

    @staticmethod
    def hamming(x, y, windowsize):
        coef = (2 * math.pi) / (windowsize - 1)
        rx, ry = 0, 0
        if abs(x) < windowsize / 2:
            rx = 0.54 + 0.46 * math.cos(coef * x)
        if abs(y) < windowsize / 2:
            ry = 0.54 + 0.46 * math.cos(coef * y)
        return rx * ry


def noise_synthesis(input_noise, mask, T, threshold, step, min_num_samples, dT=8, dstep=1):
    assert (input_noise.ndim == 2 and mask.dtype == np.bool)
    width, height = input_noise.shape
    if T == 0:
        T = min(width, height)

    print(f'Initial T= {T}, step = {step}')
    T_optimal, step_optimal = find_T_step(mask, T, threshold, step, dT=dT, dstep=dstep, min_num_samples=min_num_samples)
    # T_optimal, step_optimal = find_T_step_v1(mask, T, threshold, step, dT=dT, dstep=dstep, min_num_samples=min_num_samples)
    print(f'Optimal T= {T_optimal}, step = {step_optimal}')

    input_image = mean_inpainting(input_noise, mask)

    psd = estimate_local_psd(input_image, mask, T=T_optimal, threshold=threshold, step=step_optimal)
    noise = random_phase_noise_mosaic(psd, (width, height), alpha=0.05)

    return noise, Noise(psd)


def noise_synthesis_pca(input_noise, mask, T, threshold, step, min_num_samples, dT=8, dstep=1):
    assert(input_noise.ndim == 3 and mask.dtype == np.bool)
    width, height, nc = input_noise.shape
    if T == 0:
        T = min(width, height)

    masked_image = input_noise[mask]
    model = PCA(n_components=nc)
    masked_img_pca = model.fit_transform(masked_image)

    input_image = np.zeros((width, height, nc))
    input_image[:] = np.average(masked_img_pca, axis=0)  # should always be zero because of PCA decomposition
    input_image[mask] = masked_img_pca

    normalizer = Normalizer(input_image)
    input_image = normalizer.normalize(input_image)

    print(f'Initial T= {T}, step = {step}')
    T_optimal, step_optimal = find_T_step(mask, T, threshold, step, dT=dT, dstep=dstep, min_num_samples=min_num_samples)
    # T_optimal, step_optimal = find_T_step_v1(mask, T, threshold, step, dT=dT, dstep=dstep, min_num_samples=min_num_samples)
    print(f'Optimal T= {T_optimal}, step = {step_optimal}')

    noise_pca = []
    psds = []
    for idx_ch in range(nc):
        psd = estimate_local_psd(input_image[:, :, idx_ch], mask, T=T_optimal, threshold=threshold, step=step_optimal)
        noise_pca_per_chan = random_phase_noise_mosaic(psd, (width, height), alpha=0.05)
        noise_pca.append(noise_pca_per_chan)
        psds.append(psd)

    noise_pca = np.stack(noise_pca, axis=2)
    noise_pca = normalizer.denormalize(noise_pca)
    noise = model.inverse_transform(noise_pca.reshape((-1, nc))).reshape((width, height, nc))

    return noise, NoiseColored(psds, model, normalizer)


def noise_synthesis_with_hole_filling(input_noise, mask, method, gabor, args=None):
    assert(input_noise.ndim == 2 and mask.dtype == np.bool)
    assert method in ['patchmatch', 'opencv', 'mean']

    width, height = input_noise.shape
    T = min(width, height)

    normalizer = Normalizer(input_noise, mask=mask)
    normalized_input_noise = normalizer.normalize(input_noise, mask)

    if method == 'patchmatch':
        recon_noise = patchmatch_inpainting(normalized_input_noise, mask, erode_ratio=args['patchmatch_erode_ratio'],
                                            searchvoteiters=100, patchmatchiters=100, extrapass3x3=1)
    elif method == 'opencv':
        recon_noise = opencv_inpainting(normalized_input_noise, mask, inpaint_radius=args['opencv_inpaint_radius'],
                                        method=cv2.INPAINT_TELEA)
    elif method == 'mean':
        recon_noise = mean_inpainting(normalized_input_noise, mask)
    else:
        raise NotImplementedError

    if gabor:
        psd = gabor_approximation(recon_noise)
    else:
        step = 7  # 64 samples
        placeholder = np.ones((width, height), dtype=np.bool)
        psd = estimate_local_psd(recon_noise, placeholder, T=T-step, threshold=1, step=1)

    alpha = 0.05 if not gabor else 0.25
    op = np.real if gabor else np.abs
    noise = random_phase_noise_mosaic(psd, (width, height), alpha=alpha, op=op)

    noise = normalizer.denormalize(noise)

    return noise, Noise(psd, normalizer, alpha=alpha, op=op), normalizer.denormalize(recon_noise)


def noise_synthesis_pca_with_hole_filling(input_noise, mask, method, gabor, args=None):
    assert(input_noise.ndim == 3 and mask.dtype == np.bool)
    assert method in ['patchmatch', 'opencv', 'mean']

    width, height, nc = input_noise.shape
    T = min(width, height)

    prenormalizer = Normalizer(input_noise)
    normalized_input_noise = prenormalizer.normalize(input_noise)

    if method == 'patchmatch':
        recon_noise = patchmatch_inpainting(normalized_input_noise, mask, erode_ratio=args['patchmatch_erode_ratio'],
                                            searchvoteiters=100, patchmatchiters=100, extrapass3x3=1)
    elif method == 'opencv':
        recon_noise = opencv_inpainting(normalized_input_noise, mask, inpaint_radius=args['opencv_inpaint_radius'],
                                        method=cv2.INPAINT_TELEA)
    elif method == 'mean':
        recon_noise = mean_inpainting(normalized_input_noise, mask)
    else:
        raise NotImplementedError

    model = PCA(n_components=nc)
    input_image = model.fit_transform(recon_noise.reshape((-1, nc))).reshape((width, height, nc))

    normalizer = Normalizer(input_image)
    input_image = normalizer.normalize(input_image)

    step = 7  # 64 samples
    alpha = 0.05 if not gabor else 0.25
    op = np.real if gabor else np.abs

    noise_pca = []
    psds = []

    for c in range(nc):
        if gabor:
            psd = gabor_approximation(input_image[:, :, c])
        else:
            placeholder = np.ones((width, height), dtype=np.bool)
            psd = estimate_local_psd(input_image[:, :, c], placeholder, T=T-step, threshold=1, step=1)

        noise_pca_per_chan = random_phase_noise_mosaic(psd, (width, height), alpha=alpha, op=op)
        noise_pca.append(noise_pca_per_chan)
        psds.append(psd)

    noise_pca = np.stack(noise_pca, axis=2)
    noise_pca = normalizer.denormalize(noise_pca)
    noise = model.inverse_transform(noise_pca.reshape((-1, nc))).reshape((width, height, nc))

    noise = prenormalizer.denormalize(noise)

    return noise, NoiseColored(psds, model, normalizer, prenormalizer, alpha=alpha, op=op), prenormalizer.denormalize(recon_noise)


def opencv_inpainting(image, mask, inpaint_radius=0.1, method=cv2.INPAINT_TELEA):
    width, height = image.shape[:2]
    image_uint8 = (image * 255.0).astype(np.uint8)
    inv_mask = np.logical_not(mask).astype(np.uint8)

    inpaint_radius = int(inpaint_radius * max(width, height))
    print(f'OpenCV Inpaint: inpaintRadius = {inpaint_radius}')
    recon = cv2.inpaint(image_uint8, inv_mask, inpaint_radius, method)

    return recon / 255.0


def mean_inpainting(image, mask):
    recon = image.copy()
    recon[1 - mask] = recon[mask].mean()
    return recon


# enumerating
def find_T_step(mask, max_T, threshold, max_step, dT, dstep, min_num_samples):
    T, step = max_T, max_step
    while T >= dT:
        step = max_step
        while step >= 1:
            n_valid_samples = compute_valid_samples(mask, T, threshold, step)
            if n_valid_samples >= min_num_samples:
                return T, step
            step -= dstep

        # if still failed to find a proper T, consider T in (1, dT)
        if T == dT:
            dT = 1

        T -= dT

    # cannot reach here
    raise RuntimeError("Cannot find feasible T and step to estimate PSD, try to relax threshold")


# fine-grained enumerating
def find_T_step_v1(mask, max_T, threshold, max_step, dT, dstep, min_num_samples):
    optimal_T, optimal_step = find_T_step(mask, max_T, threshold, max_step, dT, dstep, min_num_samples)

    # fine-grained search for maximum valid T in (T, T + dT)
    T = optimal_T + 1
    while T < T + dT:
        step = max_step
        while step >= 1:
            n_valid_samples = compute_valid_samples(mask, T, threshold, step)
            if n_valid_samples > min_num_samples:
                break
            step -= dstep
        if step > 0:
            optimal_T = T
            optimal_step = step
        else:
            return optimal_T, optimal_step
        T += 1

    # cannot reach here
    raise RuntimeError("Cannot find feasible T and step to estimate PSD, try to relax threshold")


# decompose an image with binary mask into a list of procedural noise maps
def decompose(image, binary_mask, cfg, is_mask_refined, output_path):
    assert binary_mask.dtype == np.bool
    print("Now executing decompose_image func...")
    img_width, img_height = image.shape[:2]

    mask = binary_mask if is_mask_refined else erode_mask(binary_mask, kernel_size=int(cfg.erode_ratio*max(img_height, img_width)))

    multilevel_noises, base_color, remained_noise = progressive_filtering(image, mask, cfg)

    methods = cfg.noise_estimators[:len(multilevel_noises)]
    if not cfg.ignore_last:
        multilevel_noises.append(remained_noise)
        methods.append(cfg.last_noise_estimator)

    timer = Timer()

    n_levels = len(multilevel_noises)
    multilevel_syn_noises = []
    noise_models = []

    local_psd_method = noise_synthesis if image.ndim == 2 else noise_synthesis_pca
    full_psd_method = noise_synthesis_with_hole_filling if image.ndim == 2 else noise_synthesis_pca_with_hole_filling

    for i in range(n_levels):
        timer.begin(f'Synthesizing level {i}')
        methods[i].print()

        # synthesize by local PSD
        if methods[i].type == 'local':
            res = local_psd_method(multilevel_noises[i], mask, methods[i].T, methods[i].percentage,
                                   methods[i].step, methods[i].min_num_samples)

            syn_noise, noise_model = res
            multilevel_syn_noises.append(syn_noise)
            noise_models.append(noise_model)

        # synthesize using full PSD on inpainted image
        else:
            res = full_psd_method(multilevel_noises[i], mask, methods[i].method, methods[i].gabor, methods[i].params)

            syn_noise, noise_model, recon_noise = res
            multilevel_syn_noises.append(syn_noise)
            noise_models.append(noise_model)

            save_image(shift_image(recon_noise), pth.join(output_path, f'recon{i}.png'))

        timer.end(f'Noise synthesis for level {i} finished in ')

    # histogram matching for non-Gaussian noises
    multichannel = True if image.ndim == 3 else False
    placeholder = np.ones((img_width, img_height), dtype=np.bool)
    for i in range(n_levels):
        multilevel_syn_noises[i] = match_histograms(multilevel_syn_noises[i], multilevel_noises[i], placeholder, mask, multichannel)
        noise_models[i] = HistogramMatching(noise_models[i], multilevel_noises[i], mask, multichannel)

    visualize_noises(image, mask, base_color, multilevel_noises, multilevel_syn_noises, output_path)

    return multilevel_syn_noises, base_color, ProceduralMaps(noise_models, base_color)


def erode_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)
    num_valid_pixels = np.sum(eroded)
    if num_valid_pixels <= 0:
        raise RuntimeError('No pixel left after erosion.')
    return eroded


def progressive_filtering(image, mask, cfg):
    multilevel_noises = []

    cur_image = image*mask if image.ndim == 2 else image*mask[..., np.newaxis]
    cur_iter = 0

    while cur_iter < min(cfg.n_iters, len(cfg.pre_ksize)):
        img_var = masked_variance(cur_image, mask)
        if img_var < cfg.var_threshold:
            break

        print(f'At iter {cur_iter},  var = {img_var}; filtering with kernal size = {cfg.pre_ksize[cur_iter]}')

        filtered = gaussian_filter_with_mask(cur_image, mask, ksize=cfg.pre_ksize[cur_iter], sigma=0)
        noise = cur_image - filtered
        cur_image = filtered

        multilevel_noises.append(noise)
        cur_iter += 1

    img_var = masked_variance(cur_image, mask)
    print(f'On the final image: var = {img_var}')

    base_color = np.average(cur_image[mask], axis=0)
    if image.ndim == 3:
        remained_noise = cur_image - mask[..., np.newaxis] * base_color
    else:
        remained_noise = cur_image - mask * base_color

    return multilevel_noises, base_color, remained_noise


def visualize_noises(image, mask, base_color, multilevel_noises, multilevel_syn_noises, output_path):
    def dummy(x):
        return x

    n_levels = len(multilevel_noises)
    if image.ndim == 3:
        mask = np.expand_dims(mask, axis=2)
        process = shift_image
    else:
        process = dummy

    masked_image = image * mask

    tmp = masked_image.copy()
    save_image(tmp, pth.join(output_path, 'masked_img.png'))
    for i in range(n_levels):
        save_image(shift_image(multilevel_noises[i]), pth.join(output_path, "noise" + str(i) + ".png"))
        save_image(shift_image(multilevel_syn_noises[i]), pth.join(output_path, "syn_noise" + str(i) + ".png"))
        save_image(shift_image(multilevel_syn_noises[i]*mask), pth.join(output_path, "masked_syn_noise" + str(i) + ".png"))
        tmp -= multilevel_noises[i]
        save_image(np.clip(tmp, 0, 1), pth.join(output_path, "struct" + str(i) + ".png"))

    if n_levels == 0:
        pass
    elif n_levels == 1:
        fig, ax = plt.subplots(3)
        tmp = masked_image.copy()
        ax[0].imshow(tmp, cmap='gray')
        ax[1].imshow(process(multilevel_noises[0]), cmap='gray')
        tmp -= multilevel_noises[0]
        ax[2].imshow(tmp, cmap='gray')
        ax[0].set_title('masked_image')
        ax[1].set_title('filtered_noise')
        ax[2].set_title('remain_structure')
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.savefig(pth.join(output_path, 'multilevel_noises.png'), dpi=400)
        plt.close()

    else:
        fig, ax = plt.subplots(3, n_levels)
        tmp = masked_image.copy()
        for i in range(n_levels):
            ax[0, i].imshow(tmp, cmap='gray')
            ax[1, i].imshow(process(multilevel_noises[i]), cmap='gray')
            tmp -= multilevel_noises[i]
            ax[2, i].imshow(tmp, cmap='gray')
            ax[0, i].set_title(f'masked_image{i}')
            ax[1, i].set_title(f'filtered_noise{i}')
            ax[2, i].set_title(f'remain_structure{i}')
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.savefig(pth.join(output_path, 'multilevel_noises.png'), dpi=400)
        plt.close()

    # visualize
    if n_levels > 0:
        noise_image = masked_image - base_color * mask
        fig, ax = plt.subplots(3, n_levels + 1)
        ax[0, 0].imshow(masked_image, cmap='gray')
        ax[1, 0].imshow(mask * base_color, cmap='gray')
        ax[2, 0].imshow(process(noise_image), cmap='gray')
        ax[0, 0].set_title('masked_image')
        ax[1, 0].set_title('basecolor')
        ax[2, 0].set_title('noise_image')
        for i in range(n_levels):
            if masked_image.ndim == 3:
                ax[0, i + 1].imshow(shift_image(multilevel_noises[i]))
                ax[1, i + 1].imshow(shift_image(multilevel_syn_noises[i]))
                ax[2, i + 1].imshow(shift_image(multilevel_syn_noises[i]*mask))
            else:
                masked_syn_noise = multilevel_syn_noises[i]*mask
                min_ = min(np.min(multilevel_noises[i]), np.min(multilevel_syn_noises[i]), np.min(masked_syn_noise))
                max_ = max(np.max(multilevel_noises[i]), np.max(multilevel_syn_noises[i]), np.max(masked_syn_noise))
                ax[0, i + 1].imshow(multilevel_noises[i], cmap='gray', vmin=min_, vmax=max_)
                ax[1, i + 1].imshow(multilevel_syn_noises[i], cmap='gray', vmin=min_, vmax=max_)
                ax[2, i + 1].imshow(masked_syn_noise, cmap='gray', vmin=min_, vmax=max_)
            ax[0, i + 1].set_title(f'noise{i}')
            ax[1, i + 1].set_title(f'syn_noise{i}')
            ax[2, i + 1].set_title(f'masked_syn_noise{i}')
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.savefig(pth.join(output_path, 'noises_cmp.png'), dpi=400)
        plt.close()
