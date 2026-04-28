import os
from .dataset_RGB import DataLoaderDSVTrain, DataLoaderDSVVal, DataLoaderTrain, DataLoaderVal


def get_training_data(rgb_dir, img_options, dataset='sd7k'):
    assert os.path.exists(rgb_dir)
    if dataset.lower() == 'dsv':
        return DataLoaderDSVTrain(rgb_dir, img_options)
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options, dataset='sd7k'):
    assert os.path.exists(rgb_dir)
    if dataset.lower() == 'dsv':
        return DataLoaderDSVVal(rgb_dir, img_options)
    return DataLoaderVal(rgb_dir, img_options)
