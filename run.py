#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
import torch
[sys.path.append(i) for i in ['.', '..']]
# print(sys.path)

"""Wrapper to train and test a video classification model."""
from utils.parser import load_config, parse_args
from utils.misc import launch_job

from train_net import train
from test_net import test
from utils.env import setup_environment
from models.adapters.lora import LoRA_ViT_timm
from models.ast import ASTModel 

def main():
    """
    Main function to spawn the train and test process.
    """
    print(">>>>> CUDA GPU >>>>>", torch.cuda.is_available())
    setup_environment()
    args = parse_args()
    cfg = load_config(args)

    ''' LORA LAYER 아키텍처 테스트 '''
    # print("### LORA TEST ###")
    # model = ASTModel(cfg)
    # print(model)
    # lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    # print("######################")
    # print(lora_vit)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    setup_environment()
    main()
