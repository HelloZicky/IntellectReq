import torch
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
# from util import metrics
from util import new_metrics
from util.uncertainty_utils import l2_regularisation
# from loader import sequence_dataloader
# from loader import new_meta_sequence_dataloader as meta_sequence_dataloader
# from loader import new_meta_sequence_dataloader2 as meta_sequence_dataloader
from loader import multi_metric_meta_ood_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile
from tqdm import tqdm
from copy import deepcopy


# def predict(checkpoint_dir, dataset_dir):
def predict(checkpoint_dir, checkpoint_file, predict_dataset):
    device = env.get_device()
    model_obj = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
    output_file = os.path.join(checkpoint_dir, "test.txt")
    
    print(model_obj)
    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200
    WRITING_BATCH_SIZE = 512
    pred_list = []
    pred_label_list = []
    y_list = []
    ood_pred_list = []
    ood_y_list = []

    buffer = []
    _request_num = 0
    _total_num = 0
    user_id_list = []
    for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
        # logits, ood_logits, request_num, total_num = model_obj({
        # logits = model_obj({
        #     key: value.to(device)
        #     for key, value in batch_data.items()
        #     if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        #     # }, pred=True, mis_rec_threshold=mis_rec_threshold)
        # }, pred=True, uncertainty_threshold=args.uncertainty_threshold,
        #     mis_rec_threshold=args.mis_rec_threshold,
        #     stage=1, use_uncertainty_net=False)
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            # }, pred=True, mis_rec_threshold=mis_rec_threshold)
        })
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        # ndcg = 0
        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        # _request_num += request_num
        # _total_num += total_num

        buffer.extend(
            # [str(user_id), float(score), float(label)]
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )

    overall_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    # _threshold = thresholds[len(thresholds) // 2]
    # _threshold = 0.5
    # pred_label = torch.where(torch.Tensor(pred_list).view(-1, 1) >= args.mis_rec_label_threshold, 1.0, 0.0)
    # ood_y_list = torch.where(pred_label == torch.Tensor(y_list), 1.0, 0.0)
    user_auc = new_metrics.calculate_user_auc(buffer)
    # ood_auc, ood_fpr, ood_tpr, ood_thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred_list), np.array(ood_y_list))
    # ood_auc, ood_fpr, ood_tpr, ood_thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred_list), np.array(ood_y_list))
    # _ood_threshold = ood_thresholds[len(ood_thresholds) // 2]
    # _ood_threshold = 0.5
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    train_epoch = 0
    train_step = 0
    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    # with open(os.path.join(args.checkpoint_dir, output_file), "w+") as writer:
    with open(output_file, "w+") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)


def load_data(test_file):
    num_loading_workers = 1
    worker_id = worker_count = 8
    batch_size = 1024
    pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file,
        slice_id=num_loading_workers * worker_id,
        slice_count=num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataloader,
        batch_size=batch_size,
        # batch_size=1,
        num_workers=num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False
    )
    return pred_dataloader


if __name__ == '__main__':
    checkpoint_dir = "/mnt5/lzq/MetaNetwork/MetaNetwork_RS_local_amazon_ood_seq10_from38_30u30i_gru_new_auc_vae_focal_from141/checkpoint/SIGIR2023/amazon_cds_gru4rec/meta"
    checkpoint_file = "best_auc.pkl"
    test_file = "/mnt5/lzq/MetaNetwork/data/Amazon_cds/ood_generate_dataset_tiny_10_30u30i/test.txt"
    pred_dataloader = load_data(test_file)
    predict(checkpoint_dir, checkpoint_file, pred_dataloader)
