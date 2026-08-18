import argparse
import os
import torch

import numpy as np
import torch
import random

import re
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args(trial):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    # parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    # parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    # parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--ckpt_dir_1', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--eval_from', type=str, default=None)
    # parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--ood_eval', action='store_true',
                        help='Test on the OOD set')
    # parser.add_argument('--run', type=int, default=0, help='run')
    parser.add_argument('--pnn_base_widths', type=int, default=64, help='run')

    # parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--device', type=str, default='mps')
    # parser.add_argument('--device', type=str, default='cuda')
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument('--device', type=str, default=default_device)


    # Instead of the command:
    parser.add_argument('--data_dir', type=str, default='../Data/')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('-c', '--config-file', type=str, default='configs/barlow_c100.yaml')
    parser.add_argument('--hide_progress', action='store_true', default=True)

    # trained model path
    parser.add_argument('--trained_model_dir', type=str, default='./checkpoints/lump+caplus_simsiam_seq-tinyimg/')
    # specify OOD dataset
    parser.add_argument('--ood_data_name', type=str, default='seq-mnist')
    # -------------------------------------------------

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    args.ckpt_dir = f"./checkpoints/lump+{args.model.clustering}_{args.model.name}_{args.dataset.name}/"

    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval:
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]
    args.cl_type = 'scl' if args.cl_default else 'ucl'
    if args.pnn_base_widths != 64:
        pnn = '_pnn%d'%args.pnn_base_widths
    else:
        pnn = ''



    args.run = trial  # current trial index
    args.seed = trial

    args.name = args.cl_type + '_' + args.name + '_' + args.model.cl_model + '_seed_' + str(args.run)


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + datetime.now().strftime('%m%d%H%M%S_') + args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.trained_model_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'image_size': args.dataset.image_size,
        'cl_default': args.cl_default
    }
    vars(args)['dataset_kwargs'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    # for OOD dataset
    vars(args)['dataset_kwargs_ood'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset': args.ood_data_name,
        'data_dir': args.data_dir,
        'download': args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }

    return args
