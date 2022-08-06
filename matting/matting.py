import os.path as pth
import numpy as np
import matlab.engine
from sklearn.decomposition import PCA
from skimage import color

from noise.noise import LocalSpectrum
from utils import Timer
from matting.utils import normalize


def compute_spectrum_features(image, fft_size, welch_windowsize, welchstep):
    localspectrums = LocalSpectrum(image, windowsize=fft_size)
    localspectrums.compute_local_spectrum()
    localspectrums.run_welch_algorithm(welch_windowsize, welchstep)
    return localspectrums.spectrum_array


def svbrdf_matting(image, user_inputs, spectra_weight, data_path=None):
    timer = Timer()

    if user_inputs.shape[0] == 1:
        return np.ones((*image.shape[:2], 1), dtype=np.float64), np.ones((*image.shape[:2], 1), dtype=np.uint8)

    # do not use spectral features
    if spectra_weight <= 0.0:
        timer.begin()
        print('Running SVBRDF Matting (w/o spectral features)')

        eng = matlab.engine.start_matlab()
        eng.cd('matting')
        alpha_ml, alpha_binary_ml = eng.knn_matting_core_scribble(matlab.double(image.tolist()),
                                                                  matlab.double(user_inputs.tolist()), nargout=2)

        alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
        alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')

        eng.quit()
        timer.end('Matting computation complete in')
        return alpha, alpha_binary

    # if use spectral features
    fft_size = 8
    welch_windowsize, welchstep = fft_size, 2

    timer.begin()
    if data_path is not None:
        precomputed_data_file = pth.join(data_path, "joint_spectra_{}_{}_{}.npz".format(fft_size, welch_windowsize, welchstep))
    else:
        precomputed_data_file = None

    if precomputed_data_file is not None and pth.exists(precomputed_data_file):
        precomputed_features = np.load(precomputed_data_file)
        spectrum_features_pca = precomputed_features['spectrum_features_pca']
        print("Loaded precomputed spectrum features")
    else:
        print("Cannot find precomputed spectrum features. Estimating spectrum...")

        albedo_spectra = compute_spectrum_features(color.rgb2gray(image[..., 0:3]), fft_size, welch_windowsize, welchstep)
        height_spectra = compute_spectrum_features(image[..., 3], fft_size, welch_windowsize, welchstep)
        roughness_spectra = compute_spectrum_features(image[..., 4], fft_size, welch_windowsize, welchstep)
        spectrum_features = np.concatenate((albedo_spectra, height_spectra, roughness_spectra), axis=1)

        print('Reducing dimensionality by PCA')
        spectrum_features_pca = PCA(n_components=3).fit_transform(spectrum_features)
        spectrum_features_pca = spectrum_features_pca.reshape((*image.shape[:2], -1))
        spectrum_features_pca = normalize(spectrum_features_pca)
        print("Shape of reduced spectrum features:", spectrum_features_pca.shape)

        if precomputed_data_file is not None:
            np.savez(precomputed_data_file, spectrum_features_pca=spectrum_features_pca)
            print("Saved precomputed spectrum features")

    print("Running spectrum-based KNN-matting...")
    timer.begin()

    eng = matlab.engine.start_matlab()
    eng.cd('matting')
    alpha_ml, alpha_binary_ml = eng.knn_matting_spectra_core_scribble(matlab.double(image.tolist()),
                                                                      matlab.double(spectrum_features_pca.tolist()),
                                                                      matlab.double(user_inputs.tolist()),
                                                                      spectra_weight,
                                                                      nargout=2)

    alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
    alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')
    eng.quit()

    timer.end('Matting computation complete in')
    timer.end('Matting(including spectrum estimation) computation complete in')

    return alpha, alpha_binary


def svbrdf_matting_with_mask(image, mask, user_inputs, spectra_weight, data_path=None):
    timer = Timer()

    if user_inputs.shape[0] == 1:
        return np.ones((*image.shape[:2], 1), dtype=np.float64), np.ones((*image.shape[:2], 1), dtype=np.uint8)

    # do not use spectral features
    if spectra_weight <= 0.0:
        timer.begin()
        print('Running SVBRDF Matting (w/o spectral features)')

        eng = matlab.engine.start_matlab()
        eng.cd('matting')
        alpha_ml, alpha_binary_ml = eng.knn_matting_core_masked(matlab.double(image.tolist()),
                                                                matlab.logical(mask.tolist()),
                                                                matlab.double(user_inputs.tolist()),
                                                                nargout=2)

        alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
        alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')

        eng.quit()

        timer.end('Matting computation complete in')
        return alpha, alpha_binary

    # if use spectral features
    fft_size = 8
    welch_windowsize, welchstep = fft_size, 2

    timer.begin()
    if data_path is not None:
        precomputed_data_file = pth.join(data_path, "joint_spectra_{}_{}_{}.npz".format(fft_size, welch_windowsize, welchstep))
    else:
        precomputed_data_file = None

    if precomputed_data_file is not None and pth.exists(precomputed_data_file):
        precomputed_features = np.load(precomputed_data_file)
        spectrum_features_pca = precomputed_features['spectrum_features_pca']
        print("Loaded precomputed spectrum features")
    else:
        print("Cannot find precomputed spectrum features. Estimating spectrum...")

        albedo_spectra = compute_spectrum_features(color.rgb2gray(image[..., 0:3]), fft_size, welch_windowsize, welchstep)
        height_spectra = compute_spectrum_features(image[..., 3], fft_size, welch_windowsize, welchstep)
        roughness_spectra = compute_spectrum_features(image[..., 4], fft_size, welch_windowsize, welchstep)
        spectrum_features = np.concatenate((albedo_spectra, height_spectra, roughness_spectra), axis=1)

        print('Reducing dimensionality by PCA')
        spectrum_features_pca = PCA(n_components=3).fit_transform(spectrum_features)
        spectrum_features_pca = spectrum_features_pca.reshape((*image.shape[:2], -1))
        spectrum_features_pca = normalize(spectrum_features_pca)
        print("Shape of reduced spectrum features:", spectrum_features_pca.shape)

        if precomputed_data_file is not None:
            np.savez(precomputed_data_file, spectrum_features_pca=spectrum_features_pca)
            print("Saved precomputed spectrum features")

    print("Running spectrum-based KNN-matting...")
    timer.begin()

    eng = matlab.engine.start_matlab()
    eng.cd('matting')
    alpha_ml, alpha_binary_ml = eng.knn_matting_spectra_core_masked(matlab.double(image.tolist()),
                                                                    matlab.logical(mask.tolist()),
                                                                    matlab.double(spectrum_features_pca.tolist()),
                                                                    matlab.double(user_inputs.tolist()),
                                                                    spectra_weight,
                                                                    nargout=2)

    alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
    alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')
    eng.quit()

    timer.end('Matting computation complete in')
    timer.end('Matting(including spectrum estimation) computation complete in')

    return alpha, alpha_binary


def refined_matting(image, mask, user_inputs, spectra_weight, data_path=None, idx=0):
    timer = Timer()

    if user_inputs.shape[0] != 2:
        return None, None

    # do not use spectral features
    if spectra_weight <= 0.0:
        timer.begin()
        print('Running Refined Matting (w/o spectral features)')

        eng = matlab.engine.start_matlab()
        eng.cd('matting')
        alpha_ml, alpha_binary_ml = eng.knn_matting_core_masked(matlab.double(image.tolist()),
                                                                matlab.logical(mask.tolist()),
                                                                matlab.double(user_inputs.tolist()),
                                                                nargout=2)

        alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
        alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')

        eng.quit()

        timer.end('Matting computation complete in')
        return alpha, alpha_binary

    # if use spectral features
    fft_size = 8
    welch_windowsize, welchstep = fft_size, 2

    timer.begin()
    if data_path is not None:
        precomputed_data_file = pth.join(data_path, "refined_spectra_{}_{}_{}_{}.npz".format(idx, fft_size, welch_windowsize, welchstep))
    else:
        precomputed_data_file = None

    if precomputed_data_file is not None and pth.exists(precomputed_data_file):
        precomputed_features = np.load(precomputed_data_file)
        spectrum_features_pca = precomputed_features['spectrum_features_pca']
        print("Loaded precomputed spectrum features")
    else:
        print("Cannot find precomputed spectrum features. Estimating spectrum...")

        spectrum_features = compute_spectrum_features(color.rgb2gray(image), fft_size, welch_windowsize, welchstep)

        print('Reducing dimensionality by PCA')
        spectrum_features_pca = PCA(n_components=3).fit_transform(spectrum_features)
        spectrum_features_pca = spectrum_features_pca.reshape((*image.shape[:2], -1))
        spectrum_features_pca = normalize(spectrum_features_pca)
        print("Shape of reduced spectrum features:", spectrum_features_pca.shape)

        if precomputed_data_file is not None:
            np.savez(precomputed_data_file, spectrum_features_pca=spectrum_features_pca)
            print("Saved precomputed spectrum features")

    print("Running spectrum-based KNN-matting...")
    timer.begin()

    eng = matlab.engine.start_matlab()
    eng.cd('matting')
    alpha_ml, alpha_binary_ml = eng.knn_matting_spectra_core_masked(matlab.double(image.tolist()),
                                                                    matlab.logical(mask.tolist()),
                                                                    matlab.double(spectrum_features_pca.tolist()),
                                                                    matlab.double(user_inputs.tolist()),
                                                                    spectra_weight,
                                                                    nargout=2)
    alpha = np.array(alpha_ml._data).reshape(alpha_ml.size, order='F')
    alpha_binary = np.array(alpha_binary_ml._data).reshape(alpha_binary_ml.size, order='F')
    eng.quit()

    timer.end('Matting computation complete in')
    timer.end('Matting(including spectrum estimation) computation complete in')

    return alpha, alpha_binary