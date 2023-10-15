import os
import time
import json
import logging
import math
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from util.timer import Timer
# from util import oss_io
from util import args_processing as ap
from util import consts
from util import path
from util import env
from util import metrics
# from loader import sequence_dataloader
from loader import new_meta_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile


def load_model(args, model_obj):
    ckpt_path = os.path.join(args.checkpoint_dir, "best_auc.pkl")
    # output_file = os.path.join(args.checkpoint_dir, "test.txt")
    # feature_file = os.path.join(args.checkpoint_dir, "feature.pt")

    model_load = torch.load(ckpt_path)
    model_load_name_set = set()
    for name, parms in model_load.named_parameters():
        model_load_name_set.add(name)
    model_load_dict = model_load.state_dict()
    model_obj_dict = model_obj.state_dict()
    model_obj_dict.update(model_load_dict)
    model_obj.load_state_dict(model_obj_dict)
    # for name, parms in model_obj.named_parameters():
    #     if name in model_load_name_set:
    #         # print("-" * 50)
    #         # print(name)
    #         # print(parms)
    #         parms.requires_grad = False
    for name_and_parms in zip(model_obj.named_parameters(), model_load_dict.named_parameters()):
        # if name in model_load_name_set:
        #     # print("-" * 50)
        #     # print(name)
        #     # print(parms)
        #     parms.requires_grad = False
        print("=" * 50)
        print(name_and_parms[0])
        print(name_and_parms[1].size())
        print(name_and_parms[2])
        print(name_and_parms[3].size())
        if name_and_parms[1] != name_and_parms[3]:
            print("-" * 50)
            print(name_and_parms[0])
            print(name_and_parms[2])

    print(len(model_load.state_dict()))
    print(len(model_obj.state_dict()))
    return model_obj, model_load_name_set


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


args = parse_args()
ap.print_arguments(args)
# args.checkpoint_dir = "../../../checkpoint/NIPS2022/amazon_beauty_din/meta"
# args.model = "meta_din"

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta
# model_meta = model.get_model_meta("meta_din")  # type: model.ModelMeta
model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)
model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
model_obj, model_load_name_set = load_model(args, model_obj)