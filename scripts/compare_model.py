import math
import os
import numpy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
import model


ROOT_FOLDER = "../checkpoint/SIGIR2023"
DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty"]

# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]
#
# TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]

MODEL_LIST = ["din", "sasrec", "gru4rec"]

# log_filename = "best_auc.pkl"
# log_filename_ = "stage3_best_auc_ood.pkl"

for dataset in DATASET_LIST:
    print("{}{}{}".format("=" * 50, dataset, "=" * 50))
    for model in MODEL_LIST:
        print("{}{}{}".format("-" * 20, model, "-" * 20))

        # model_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "base", "best_auc.pkl")
        model_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta", "best_auc.pkl")
        # model_file = "/data/lvzheqi/MetaNetwork/MetaNetwork_RS_local_amazon_ood_seq10_from38_30u30i_gru_new_auc_vae_focal_from141_seed0_newuncertainty/checkpoint/SIGIR2023/amazon_cds_din/meta/best_auc.pkl"
        model_uncertainty_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta_ood_uncertainty5", "stage3_best_auc_ood.pkl")

        # model_dict = torch.load(model_file).state_dict()
        print(os.path.exists(model_file))
        model_dict = torch.load(model_file)
        model_uncertainty_dict = torch.load(model_uncertainty_file).state_dict()

        # count1 = 0
        # count2 = 0
        print("-" * 50)
        for key, value in model_dict.items():
            print(key)
            if model_dict[key] == model_uncertainty_dict[key]:
                print("=")
            elif model_dict[key] != model_uncertainty_dict[key]:
                print("!=")
