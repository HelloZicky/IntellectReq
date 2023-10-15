import sys
sys.path.append("../")
import model
from util import consts
from thop import profile
from util import args_processing as ap
import os
import torch
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

prefix = "/mnt3/lzq/MetaNetwork/MetaNetwork_RS/scripts/"
# postfix = "base/movielens_conf.json"
postfix = "base/movielens_100k_conf.json"

device = "cuda:0"
for model_name in ["din", "sasrec", "gru4rec"]:
    model_meta = model.get_model_meta(model_name)  # type: model.ModelMeta
    if model_name == "din":
        model_name = "DIN"
    arch_config = os.path.join(prefix, "new_" + model_name, postfix)
    # Load model configuration
    model_conf, raw_model_conf = ap.parse_arch_config_from_args_get_profile(model_meta, arch_config, bucket=None)  # type: dict
    model_obj = model_meta.model_builder(model_conf=model_conf).to(device)  # type: torch.nn.module

    # logits = model_obj({
    #             key: value.to(device)
    #             for key, value in batch_data.items()
    #             if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
    # })
    batch_data = {
        consts.FIELD_USER_ID: torch.LongTensor([0]),
        # consts.FIELD_USER_ID: torch.randint(100, (1, 1)),
        # consts.FIELD_TARGET_ID: torch.Tensor([0]),
        # consts.FIELD_TARGET_ID: torch.randint(100, (1, 1)),
        consts.FIELD_TARGET_ID: torch.LongTensor([0]),
        # consts.FIELD_CLK_SEQUENCE: torch.from_numpy(torch.randn(30)),
        consts.FIELD_CLK_SEQUENCE: torch.randint(100, (1, 30)),
        # consts.FIELD_LABEL: torch.from_numpy(np.stack([item[3] for item in data], axis=0))
        # consts.FIELD_LABEL: torch.randint(1, (1, 1))
        consts.FIELD_LABEL: torch.LongTensor([0]),
    }
    flops, params = profile(model_obj, inputs=({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
    },))
    print(model)
    # print("flops={}B, params={}M".format(flops / 1e9, params / 1e6))
    print("flops={}M, params={}K".format(flops / 1e6, params / 1e3))