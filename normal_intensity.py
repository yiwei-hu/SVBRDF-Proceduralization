import warnings


# pre-defined normal intensity
def get_normal_intensity(dataset_name):
    if dataset_name in ['ground', 'mosaic', 'metal', 'painted_plaster']:
        return 1
    elif dataset_name == 'stone_smalltile':
        return 3
    else:
        warnings.warn('No specified normal intensity. Use default value 1.')
        return 1