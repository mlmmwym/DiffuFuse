from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import torch

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from dfmodules.backbone import (
    configure_huggingface_endpoint,
    patch_backbone_xsoftmax_for_torch,
    prepare_backbone_files,
)
from dfmodules.model import DiffFuseRegression
from src.common import build_seed_list, progress_bar, set_random_seed, setup_device, str2bool
from src.data import build_data_loaders
from src.training import (
    build_optimizer_and_scheduler,
    train_epoch,
)
from src.utils.evaluation import compute_selection_score, evaluate, format_metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="mosei", choices=["mosi", "mosei"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--manual_mirror_download", type=str2bool, default=True)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dev_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=5217)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--seed_min", type=int, default=1000)
    parser.add_argument("--seed_max", type=int, default=10000)
    parser.add_argument("--num_random_seeds", type=int, default=5)
    parser.add_argument("--seed_sampler_seed", type=int, default=5217)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--text_index", type=int, default=3)
    parser.add_argument("--visual_index", type=int, default=1)
    parser.add_argument("--acoustic_index", type=int, default=2)
    parser.add_argument("--av_lstm_hidden_size", type=int, default=256)
    parser.add_argument("--av_lstm_num_layers", type=int, default=1)
    parser.add_argument("--cam_num_heads", type=int, default=4)
    parser.add_argument("--diffusion_beta_start", type=float, default=1e-5)
    parser.add_argument("--diffusion_beta_end", type=float, default=0.02)
    parser.add_argument("--diffusion_loss_weight", type=float, default=0.1)
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--batch_progress", type=str2bool, default=False)
    return parser


def build_model(args, visual_dim, acoustic_dim):
    return DiffFuseRegression(
        visual_dim=visual_dim,
        acoustic_dim=acoustic_dim,
        av_lstm_hidden_size=args.av_lstm_hidden_size,
        av_lstm_num_layers=args.av_lstm_num_layers,
        cam_num_heads=args.cam_num_heads,
        shared_tensor_fusion_dim=16,
        diffusion_steps=50,
        diffusion_beta_start=args.diffusion_beta_start,
        diffusion_beta_end=args.diffusion_beta_end,
        diffusion_eval_step=None,
        dropout_prob=0.2,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        backbone_path=args.backbone_path,
    )


def run_one_seed(args, seed, seed_index, num_seeds, train_loader, dev_loader, test_loader, visual_dim, acoustic_dim, device, global_best):
    _ = seed_index, num_seeds
    args.seed = seed
    set_random_seed(seed)
    model = build_model(args, visual_dim, acoustic_dim)
    patch_backbone_xsoftmax_for_torch()
    model.to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, args)
    best_selection_score = None
    best_epoch = 0
    best_test_metrics = None
    for epoch_i in range(args.n_epochs):
        epoch_desc = "Seed {}/{} Epoch {}/{} Train".format(seed_index, num_seeds, epoch_i + 1, args.n_epochs)
        dev_desc = "Seed {}/{} Epoch {}/{} Dev".format(seed_index, num_seeds, epoch_i + 1, args.n_epochs)
        test_desc = "Seed {}/{} Epoch {}/{} Test".format(seed_index, num_seeds, epoch_i + 1, args.n_epochs)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args, device, desc=epoch_desc)
        dev_metrics = evaluate(model, dev_loader, device, dev_desc, show_progress=args.batch_progress)
        selection_score = compute_selection_score(dev_metrics)
        _ = train_loss
        if best_selection_score is None or selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch_i
            best_test_metrics = evaluate(model, test_loader, device, test_desc, show_progress=args.batch_progress)
        if global_best["score"] is None or selection_score > global_best["score"]:
            global_best["score"] = selection_score
            global_best["seed"] = seed
            global_best["epoch"] = epoch_i
            global_best["dev_metrics"] = dev_metrics
            global_best["test_metrics"] = best_test_metrics

    metrics = best_test_metrics
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    args = build_parser().parse_args()
    args.hf_endpoint = "https://hf-mirror.com"
    configure_huggingface_endpoint(args)
    args.backbone_path = prepare_backbone_files(args)
    device = setup_device(args)

    seeds = build_seed_list(args)
    train_loader, dev_loader, test_loader, visual_dim, acoustic_dim = build_data_loaders(args)
    all_metrics = []
    global_best = {
        "score": None,
        "seed": None,
        "epoch": None,
        "dev_metrics": None,
        "test_metrics": None,
    }

    for seed_index, seed in enumerate(progress_bar(seeds, desc="training", leave=True, position=0), start=1):
        metrics = run_one_seed(
            args,
            seed,
            seed_index,
            len(seeds),
            train_loader,
            dev_loader,
            test_loader,
            visual_dim,
            acoustic_dim,
            device,
            global_best,
        )
        all_metrics.append(metrics)

    if all_metrics:
        if global_best["score"] is not None:
            print(
                "selection: seed={}, epoch={}, {}".format(
                    global_best["seed"],
                    global_best["epoch"] + 1,
                    format_metrics("{}_dev".format(args.dataset), global_best["dev_metrics"]),
                )
            )
            print(
                "result: {}".format(
                    format_metrics("{}_test".format(args.dataset), global_best["test_metrics"]),
                )
            )


if __name__ == "__main__":
    main()
