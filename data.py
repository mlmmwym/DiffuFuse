from __future__ import absolute_import, division, print_function

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfmodules.backbone import load_backbone_tokenizer


def load_split(data_dir, file_name):
    path = Path(data_dir) / file_name
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_label(label_id):
    if isinstance(label_id, str):
        parts = label_id.split(",")
        return float(parts[1] if len(parts) > 1 else label_id)
    return float(label_id)


def unpack_multimodal_example(example, args):
    fields, label_id, segment = example
    text = fields[args.text_index]
    visual = np.asarray(fields[args.visual_index], dtype=np.float32)
    acoustic = np.asarray(fields[args.acoustic_index], dtype=np.float32)
    if not isinstance(text, str):
        raise ValueError("Expected raw text at fields[{}] for segment {}.".format(args.text_index, segment))
    if visual.ndim != 2:
        raise ValueError("Expected 2D visual features at fields[{}], got shape {} for segment {}.".format(args.visual_index, visual.shape, segment))
    if acoustic.ndim != 2:
        raise ValueError("Expected 2D acoustic features at fields[{}], got shape {} for segment {}.".format(args.acoustic_index, acoustic.shape, segment))
    return text, visual, acoustic, parse_label(label_id)


def resample_to_length(sequence, target_length):
    if target_length <= 0:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)
    if sequence.shape[0] == target_length:
        return sequence.astype(np.float32)
    if sequence.shape[0] == 0:
        return np.zeros((target_length, sequence.shape[1]), dtype=np.float32)

    source_positions = np.linspace(0, sequence.shape[0] - 1, num=sequence.shape[0])
    target_positions = np.linspace(0, sequence.shape[0] - 1, num=target_length)
    aligned = np.empty((target_length, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        aligned[:, dim] = np.interp(target_positions, source_positions, sequence[:, dim])
    return aligned


def pad_feature_sequence(sequence, target_length):
    padded = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    valid_length = min(sequence.shape[0], target_length)
    if valid_length > 0:
        padded[:valid_length] = sequence[:valid_length]
    return padded


def inspect_feature_dims(split_data, args):
    visual_dims = set()
    acoustic_dims = set()
    for data in split_data.values():
        for example in data:
            _, visual, acoustic, _ = unpack_multimodal_example(example, args)
            visual_dims.add(visual.shape[1])
            acoustic_dims.add(acoustic.shape[1])
    if len(visual_dims) != 1 or len(acoustic_dims) != 1:
        raise ValueError("Feature dimensions are inconsistent: visual={}, acoustic={}".format(sorted(visual_dims), sorted(acoustic_dims)))
    visual_dim = visual_dims.pop()
    acoustic_dim = acoustic_dims.pop()
    print("Feature dims: visual_dim={}, acoustic_dim={}".format(visual_dim, acoustic_dim))
    return visual_dim, acoustic_dim


def convert_to_multimodal_dataset(data, tokenizer, args):
    if args.max_seq_length < 2:
        raise ValueError("--max_seq_length must be at least 2 when using token sequence fusion without [CLS].")

    texts = []
    visual_sequences = []
    acoustic_sequences = []
    labels = []
    for example in data:
        text, visual, acoustic, label = unpack_multimodal_example(example, args)
        texts.append(text)
        visual_sequences.append(visual)
        acoustic_sequences.append(acoustic)
        labels.append(label)

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt",
    )
    aligned_max_length = max(1, args.max_seq_length - 1)
    sequence_lengths = (encoded["attention_mask"].sum(dim=1) - 1).clamp(min=1, max=aligned_max_length)
    visual_features = []
    acoustic_features = []
    for visual, acoustic, sequence_length in zip(visual_sequences, acoustic_sequences, sequence_lengths.tolist()):
        aligned_visual = resample_to_length(visual, sequence_length)
        aligned_acoustic = resample_to_length(acoustic, sequence_length)
        visual_features.append(pad_feature_sequence(aligned_visual, aligned_max_length))
        acoustic_features.append(pad_feature_sequence(aligned_acoustic, aligned_max_length))

    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(np.array(visual_features), dtype=torch.float),
        torch.tensor(np.array(acoustic_features), dtype=torch.float),
        sequence_lengths.long(),
        torch.tensor(labels, dtype=torch.float),
    )


def build_data_loaders(args):
    train_file = "{}_train.pkl".format(args.dataset)
    dev_file = "{}_dev.pkl".format(args.dataset)
    test_file = "{}_test.pkl".format(args.dataset)
    split_data = {
        "train": load_split(args.data_dir, train_file),
        "dev": load_split(args.data_dir, dev_file),
        "test": load_split(args.data_dir, test_file),
    }
    visual_dim, acoustic_dim = inspect_feature_dims(split_data, args)
    tokenizer = load_backbone_tokenizer(
        backbone_path=args.backbone_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    train_dataset = convert_to_multimodal_dataset(split_data["train"], tokenizer, args)
    dev_dataset = convert_to_multimodal_dataset(split_data["dev"], tokenizer, args)
    test_dataset = convert_to_multimodal_dataset(split_data["test"], tokenizer, args)

    return (
        DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False),
        visual_dim,
        acoustic_dim,
    )
