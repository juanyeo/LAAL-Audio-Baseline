#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from fvcore.common.registry import Registry

from .esc50 import ESC50Dataset

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()
    if name.upper() == "EPICSOUNDS":
        return DATASET_REGISTRY.get(name)(cfg, split)
    elif name.upper() == "ESC50":
        if split in ["train", "train+val"]:
            return ESC50Dataset(cfg=cfg, dataset_json_file=cfg.ESC.TRAIN_DATA_FILE, label_csv=cfg.ESC.LABEL_CSV)
        else:
            return ESC50Dataset(cfg=cfg, dataset_json_file=cfg.ESC.TEST_DATA_FILE, label_csv=cfg.ESC.LABEL_CSV)
    else:
        return DATASET_REGISTRY.get(name)(cfg, split)
