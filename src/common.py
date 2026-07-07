from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


def progress_bar(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(*args, **kwargs)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_seed_list(seed_text):
    seeds = []
    for item in str(seed_text).split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    return seeds


def build_seed_list(args):
    if args.seeds:
        return parse_seed_list(args.seeds)
    if args.seed_min > args.seed_max:
        raise ValueError("--seed_min must be <= --seed_max.")
    population_size = args.seed_max - args.seed_min + 1
    if args.num_random_seeds <= 0:
        raise ValueError("--num_random_seeds must be positive.")
    if args.num_random_seeds > population_size:
        raise ValueError("--num_random_seeds cannot exceed the inclusive seed range size.")
    sampler = random.Random(args.seed_sampler_seed)
    return sampler.sample(range(args.seed_min, args.seed_max + 1), args.num_random_seeds)


def set_random_seed(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def setup_device(args):
    if args.gpu_id < 0 or not torch.cuda.is_available():
        print("Using device: cpu")
        return torch.device("cpu")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError("Requested gpu_id {}, but only {} CUDA device(s) are available.".format(args.gpu_id, torch.cuda.device_count()))
    device = torch.device("cuda:{}".format(args.gpu_id))
    torch.cuda.set_device(device)
    print("Using device: {} ({})".format(device, torch.cuda.get_device_name(args.gpu_id)))
    return device
