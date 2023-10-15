# -*- coding: utf-8 -*-
"""
Argument parsing and dumpling
"""
import copy
import json
import base64
import os.path
from argparse import Namespace

from . import path


_ARCH_CONFIG_FILE_NAME = "arch_conf.json"
_TRAIN_ARGS_FILE_NAME = "train_args.json"


def get_init_checkpoint_file(buckets, checkpoint_file, init_step):
    if checkpoint_file and init_step is not None and buckets:
        full_init_checkpoint_dir = path.join_oss_path(buckets, checkpoint_file)
        full_init_checkpoint_file = full_init_checkpoint_dir + "/model.ckpt-{}".format(init_step)
        if not gfile.Exists(full_init_checkpoint_file + ".index"):
            full_init_checkpoint_file = full_init_checkpoint_dir + "/model.ckpt-{}".format(init_step + 1)

        return full_init_checkpoint_file
    else:
        return None


# def parse_arch_config_from_args(model_meta, args, bucket):
#     """
#     Read or parse arch config
#     :param model_meta:
#     :param args:
#     :return:
#     """
#     if args.arch_config is not None:
#         raw_arch_config = json.loads(base64.b64decode(args.arch_config))
#     elif args.arch_config_path is not None:
#         with open(args.arch_config_path, "rt") as reader:
#             raw_arch_config = json.load(reader)
#     else:
#         raise KeyError("Model configuration not found")
#
#     return model_meta.arch_config_parser(raw_arch_config), raw_arch_config

def parse_arch_config_from_args(model, args):
    """
    Read or parse arch config
    :param model:
    :param args:
    :return:
    """
    if args.arch_config is not None:
        with open(args.arch_config) as jsonfile:
            raw_arch_config = json.load(jsonfile)
    elif args.arch_config_path is not None:
        with open(args.arch_config_path, "rt") as reader:
            raw_arch_config = json.load(reader)
    else:
        raise KeyError("Model configuration not found")

    return model.arch_config_parser(raw_arch_config), raw_arch_config



def parse_arch_config_from_args_get_profile(model_meta, arch_config, bucket):
    """
    Read or parse arch config
    :param model_meta:
    :param args:
    :return:
    """
    print(arch_config)
    # raw_arch_config = json.loads(arch_config)
    f = open(arch_config, encoding="utf-8")
    raw_arch_config = json.load(f)

    return model_meta.arch_config_parser(raw_arch_config), raw_arch_config


def load_arch_config(model_meta, checkpoint_dir, bucket):
    """
    Load arch config from OSS
    :param model_meta:
    :param args:
    :return:
    """
    content = bucket.get_object(path.join_oss_path(checkpoint_dir, _ARCH_CONFIG_FILE_NAME)).read()
    return model_meta.arch_config_parser(json.loads(content))


def load_config(checkpoint_dir, file_name, bucket):
    content = bucket.get_object(path.join_oss_path(checkpoint_dir, file_name)).read()
    return json.loads(content)


# def dump_config(checkpoint_dir, file_name, config_obj, bucket):
#     """
#     Dump configurations to OSS
#     :param checkpoint_dir:
#     :param file_name:
#     :param config_obj:
#     :return:
#     """
#     bucket.put_object(
#         path.join_oss_path(checkpoint_dir, file_name), json.dumps(config_obj)
#     )

def dump_config(checkpoint_dir, file_name, config_obj):
    """
    Dump configurations to OSS
    :param checkpoint_dir:
    :param file_name:
    :param config_obj:
    :return:
    """
    print(config_obj, file=open(os.path.join(checkpoint_dir, file_name), "w+"))


# def dump_model_config(checkpoint_dir, raw_model_arch, bucket):
def dump_model_config(checkpoint_dir, raw_model_arch):
    """
    Dump model configurations to OSS
    :param args: Namespace object, parsed from command-line arguments
    :param raw_model_arch:
    :return:
    """
    dump_config(checkpoint_dir, _ARCH_CONFIG_FILE_NAME, raw_model_arch)


# def dump_train_arguments(checkpoint_dir, args, bucket):
def dump_train_arguments(checkpoint_dir, args):
    args_dict = copy.copy(args.__dict__)
    args_dict.pop("arch_config")
    dump_config(checkpoint_dir, _TRAIN_ARGS_FILE_NAME, args_dict)


def load_train_arguments(checkpoint_dir, bucket):
    ns = Namespace()
    content = bucket.get_object(path.join_oss_path(checkpoint_dir, _TRAIN_ARGS_FILE_NAME)).read()
    for key, value in json.loads(content).items():
        setattr(ns, key, value)

    return ns


def print_arguments(args):
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))
