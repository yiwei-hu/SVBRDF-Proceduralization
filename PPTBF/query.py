import glob
import json
import math
import os
import pathlib

from PIL import Image
from pyflann import *
from scipy import stats

import PPTBF.utils as utils
import PPTBF.vgg as vgg


def get_image_subdir(index, image_index):
    for i in range(len(image_index) - 1):
        if image_index[i] <= index < image_index[i + 1]:
            return i + 1, index - image_index[i]

    return 0, index


def load_parameter_file(filepath):
    # extract thresholding value from the folder name
    folder_name = pathlib.PurePath(filepath).name
    param_list = folder_name.split('_')
    if param_list[1] == '0':
        thresholding = 0.8
    elif param_list[1] == '1':
        thresholding = 0.5
    else:
        thresholding = 0.2

    # load the corresponding parameter file
    filename = glob.glob(os.path.join(filepath, 'pptbf_*.txt'))
    filename = filename[0]
    parameters = np.loadtxt(filename, skiprows=1)
    print("Loading parameter file: ", filename)
    filename = pathlib.Path(filename).stem
    param_list = filename.split('_')
    tiling_type, window_shape, feature_mixture, inv_phase = param_list[2], param_list[3], param_list[4], param_list[5]

    return thresholding, int(tiling_type), int(window_shape), int(feature_mixture), int(inv_phase), parameters


def build_nearest_neighbor_arch(root, distance_type, load_index=True):
    # load extracted feature matrices
    features = utils.load_features(os.path.join(root, 'features_reduced.npy'))
    print("Feature matrix: ", features.shape)

    # build FLANN architecture to speed up query
    flann = FLANN()
    if not load_index:
        set_distance_type(distance_type)
        params = flann.build_index(features, algorithm="autotuned", target_precision=0.99, log_level="info")
        flann.save_index(os.path.join(root, 'features_index.idx'))
        with open(os.path.join(root, 'search_params.json'), 'w') as f:
            json.dump(params, f)
    else:
        flann.load_index(os.path.join(root, 'features_index.idx'), pts=features)
        with open(os.path.join(root, 'search_params.json'), 'r') as f:
            params = json.load(f)

    print(params)
    return flann, params


def query(image, num_neighbors, save_image, **kwargs):
    filename = kwargs['filename']
    output_path = kwargs['output_path']
    vgg19 = kwargs['vgg']
    weight_lbp = kwargs['weight_lbp']
    weight_fft = kwargs['weight_fft']
    ipca = kwargs['ipca']
    flann = kwargs['flann']
    params = kwargs['flann_params']
    image_index = kwargs['image_index']
    sub_dirs = kwargs['sub_dirs']

    feature = vgg.feature_vector(image=image, vgg=vgg19, weight_lbp=math.sqrt(weight_lbp), weight_fft=math.sqrt(weight_fft))
    feature_original = np.float32(feature)
    feature = ipca.transform(feature_original.reshape(1, -1))
    results, dists = flann.nn_index(feature, num_neighbors, checks=params["checks"])
    if num_neighbors > 1:
        results = results[0]
    query_parameters = np.empty([0, 29])
    for ith, index in enumerate(results):
        dir_index, offset = get_image_subdir(index, image_index)
        thresholding, tiling_type, window_shape, feature_mixture, inv_phase, parameters = load_parameter_file(sub_dirs[dir_index])
        parameter = parameters[offset, 1:]
        parameter = np.insert(parameter, [0], [thresholding, tiling_type])
        query_parameters = np.concatenate((query_parameters, parameter.reshape(1, -1)), axis=0)
        if save_image:
            image_files = utils.get_image_files(sub_dirs[dir_index])
            query_image = Image.open(image_files[offset]).convert('L')
            utils.save_image(query_image, os.path.join(output_path, filename + '_query_' + str(ith) + '.png'))

    return stats.mode(query_parameters, axis=0)[0].reshape(-1), query_parameters
