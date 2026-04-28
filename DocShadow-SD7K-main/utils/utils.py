import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = _unique_checkpoint_path(os.path.join(outdir, model_name + '_' + 'epoch_' + str(epoch) + '.pth'))
    torch.save(state, checkpoint_file)
    return checkpoint_file


def make_checkpoint_dir(base_dir, model_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return _unique_checkpoint_path(os.path.join(base_dir, model_name + '_' + timestamp), extension='')


def _unique_checkpoint_path(path, extension='.pth'):
    if not os.path.exists(path):
        return path

    if extension:
        stem = path[:-len(extension)]
        suffix = extension
    else:
        stem = path
        suffix = ''

    version = 2
    while True:
        candidate = stem + '_v' + str(version) + suffix
        if not os.path.exists(candidate):
            return candidate
        version += 1


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)
