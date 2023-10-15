# coding=utf-8
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

from util import utils

utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")
    parser.add_argument("--positive_dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--learning_rate_uncertainty", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    # parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch_uncertainty", type=int, default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--model_uncertainty", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
def predict_stage1(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0, mis_rec_threshold=0.5):
    # Load pretrained models

    # ckpt_path = path.join_oss_path(
    #     args.checkpoint_dir,
    #     consts.FILE_CHECKPOINT_PREFIX + str(args.step)
    # )
    # checkpoint = oss_io.load_checkpoint(
    #     bucket=bucket,
    #     model_path=ckpt_path
    # )
    # model_obj.load_state_dict(checkpoint[consts.STATE_MODEL])

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
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            # }, pred=True, mis_rec_threshold=mis_rec_threshold)
        }, pred=True, uncertainty_threshold=args.uncertainty_threshold,
            mis_rec_threshold=args.mis_rec_threshold,
            stage=1, use_uncertainty_net=False)
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

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "stage1_log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20

    # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
    #       "ood_auc={:5f}, request_num={}, total_num={}, _threshold={}, _ood_threshold={}".
    #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
    #              ood_auc, _request_num, _total_num, _threshold, _ood_threshold))
    # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
    #     print(
    #         "train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #         "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
    #         "ood_auc={:5f}, request_num={}, total_num={}, _threshold={}, _ood_threshold={}".
    #             format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #                    user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
    #                    ood_auc, _request_num, _total_num, _threshold, _ood_threshold),
    #         file=writer)
    # 
    # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, ood_auc, _threshold, _ood_threshold


def predict_stage2(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0, mis_rec_threshold=0.5):
    # Load pretrained models

    # ckpt_path = path.join_oss_path(
    #     args.checkpoint_dir,
    #     consts.FILE_CHECKPOINT_PREFIX + str(args.step)
    # )
    # checkpoint = oss_io.load_checkpoint(
    #     bucket=bucket,
    #     model_path=ckpt_path
    # )
    # model_obj.load_state_dict(checkpoint[consts.STATE_MODEL])

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
    buffer_uncertainty = []
    _request_num = 0
    _total_num = 0
    user_id_list = []

    pred_mse = 0
    pred_uncertainty = 0

    uncertainty_list = []
    for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
        # logits, ood_logits, request_num, total_num = model_obj({
        logits, mse, uncertainty, request_num = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            # }, pred=True, mis_rec_threshold=mis_rec_threshold)
        }, pred=True, uncertainty_threshold=args.uncertainty_threshold,
            mis_rec_threshold=args.mis_rec_threshold,
            stage=2, use_uncertainty_net=False)

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        # print("*" * 30)
        # print(logits.size())
        # print(y.size())
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

        pred_mse += torch.sum(mse)
        # pred_uncertainty += torch.sum(uncertainty)
        pred_uncertainty += 0
        _request_num += request_num

        # uncertainty_list.extend(uncertainty.detach().cpu().numpy())

        # buffer_uncertainty.extend(
        #     # [str(user_id), float(score), float(label)]
        #     [int(user_id), float(mse), float(uncertainty)]
        #     for user_id, mse, uncertainty
        #     in zip(
        #         batch_data[consts.FIELD_USER_ID],
        #         mse,
        #         uncertainty
        #     )
        # )
        torch.cuda.empty_cache()
    # print(uncertainty_list)
    # print(len(uncertainty_list))
    # args.uncertainty_threshold = torch.Tensor([sorted(uncertainty_list)[len(uncertainty_list) * 10 // 100]])
    # _total_num = len(buffer_uncertainty)
    # pred_mse = pred_mse / _total_num
    # pred_uncertainty = pred_uncertainty / _total_num

    # print("train_epoch={}, train_step={}, mse={:5f}, uncertainty={:5f}, "
    #       "request_num={}, total_num={}, _threshold={}, _ood_threshold={}".
    #       format(train_epoch, train_step, pred_mse, pred_uncertainty,
    #              _request_num, _total_num, _threshold, _ood_threshold))
    _ood_threshold = mis_rec_threshold
    # print("train_epoch={}, train_step={}, mse={:5f}, uncertainty={:5f}, "
    #       "request_num={}, total_num={}, _ood_threshold={}".
    #       format(train_epoch, train_step, pred_mse, pred_uncertainty,
    #              _request_num, _total_num, _ood_threshold))
    #
    # with open(os.path.join(args.checkpoint_dir, "stage2_log_ood.txt"), "a") as writer:
    #     print("train_epoch={}, train_step={}, mse={:5f}, uncertainty={:5f}, "
    #           "request_num={}, total_num={}, _ood_threshold={}".
    #           format(train_epoch, train_step, pred_mse, pred_uncertainty,
    #                  _request_num, _total_num, _ood_threshold),
    #           file=writer)

    overall_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))

    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "stage2_log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, \
           pred_mse, pred_uncertainty, _ood_threshold

    # return pred_mse, pred_uncertainty, _ood_threshold


def predict_stage3(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0, mis_rec_threshold=0.5):
    # Load pretrained models

    # ckpt_path = path.join_oss_path(
    #     args.checkpoint_dir,
    #     consts.FILE_CHECKPOINT_PREFIX + str(args.step)
    # )
    # checkpoint = oss_io.load_checkpoint(
    #     bucket=bucket,
    #     model_path=ckpt_path
    # )
    # model_obj.load_state_dict(checkpoint[consts.STATE_MODEL])

    model_obj.eval()
    model_obj.to(device)

    # model_obj.eval()
    # model_obj.to(device)

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
        logits, ood_logits, request_num, total_num = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            # }, pred=True, mis_rec_threshold=mis_rec_threshold)
        }, pred=True, uncertainty_threshold=args.uncertainty_threshold,
            mis_rec_threshold=args.mis_rec_threshold,
            # stage=3, use_uncertainty_net=True)
            stage=3, use_uncertainty_net=args.use_uncertainty_net)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        # ndcg = 0
        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))
        if not args.use_uncertainty_net:
            ood_prob = ood_logits.detach().cpu().numpy()
        else:
            ood_prob = torch.sigmoid(ood_logits).detach().cpu().numpy()
        pred_label = torch.where(torch.Tensor(prob).view(-1, 1) >= args.mis_rec_label_threshold, 1.0, 0.0)
        ood_y = torch.where(pred_label == y, 1.0, 0.0)
        # print("=" * 50)
        # print("ood_prob\n", ood_prob.reshape(1, -1))
        # print("ood_y\n", ood_y.view(1, -1))
        ood_pred_list.extend(ood_prob.tolist())
        ood_y_list.extend(np.array(ood_y).tolist())

        _request_num += request_num
        _total_num += total_num

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
    ood_auc, ood_fpr, ood_tpr, ood_thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred_list), np.array(ood_y_list))
    # _ood_threshold = ood_thresholds[len(ood_thresholds) // 2]
    _ood_threshold = 0.5
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
          "ood_auc={:5f}, request_num={}, total_num={}, args.mis_rec_label_threshold={}, args.mis_rec_threshold={}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
                 ood_auc, _request_num, _total_num, args.mis_rec_label_threshold, args.mis_rec_threshold))
    # with open(os.path.join(args.checkpoint_dir, "stage3_log_ood.txt"), "a") as writer:
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print(
            "train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
            "ood_auc={:5f}, request_num={}, total_num={}, args.mis_rec_label_threshold={}, args.mis_rec_threshold={}".
                format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                       user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
                       ood_auc, _request_num, _total_num, args.mis_rec_label_threshold, args.mis_rec_threshold),
            file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, ood_auc, args.mis_rec_label_threshold, args.mis_rec_threshold


# def train(train_dataset, model_obj, device, args, bucket, pred_dataloader):
def train_stage1(positive_train_dataset, train_dataset, model_uncertainty_obj, model_obj, device, args,
                 positive_pred_dataloader, pred_dataloader, model_load_name_set=None, best_model=None):
    if best_model is not None:
        # model_uncertainty_obj = deepcopy(best_model)
        model_uncertainty_obj = best_model
    criterion = nn.BCEWithLogitsLoss()
    optimizer_uncertainty = torch.optim.Adam(
        model_uncertainty_obj.parameters(),
        lr=float(args.learning_rate)
    )
    # optimizer = torch.optim.Adam(
    #     model_obj.parameters(),
    #     lr=float(args.learning_rate)
    # )

    model_uncertainty_obj.train()
    model_uncertainty_obj.to(device)
    print(model_uncertainty_obj)

    # model_obj.train()
    # model_obj.to(device)
    # print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200

    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "stage1_best_auc" + ".pkl")

    best_ood_auc = 0

    best_ckpt_path = os.path.join(args.checkpoint_dir, "stage1_best_auc" + "_ood.pkl")

    # train_threshold = 0.5
    # train_ood_threshold = 0.2

    # stage1_layer_list = []
    # for name, parms in model_uncertainty_obj.named_parameters():
    #     args.stage1_layer_list.append(name)

    zero_grad_dict = {}
    # stage2_layer_list = []
    all_layers_list = []
    fix_layer = args.stage1_fix_layer
    if fix_layer:
        for name, parms in model_uncertainty_obj.named_parameters():
            # if name in args.stage1_layer_list:
            all_layers_list.append(name)
            if name.split("_")[0] != "stage2" and name.split("_")[0] != "stage3":
                args.stage1_layer_list.append(name)
                # print("-" * 50)
                # print(name)
                # print(parms)
                parms.requires_grad = True
            else:
                parms.requires_grad = False
                zero_grad_dict[name] = torch.zeros_like(parms).to(device)

        # print("all_layers_list ", all_layers_list)
        # print("args.stage1_layer_list ", args.stage1_layer_list)

    ### stage1: RecSys model and Mis-recommendation model
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            # logits, ood_logits = model_obj({
            logits = model_uncertainty_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                # }, train_ood_threshold=train_ood_threshold)
            }, uncertainty_threshold=args.uncertainty_threshold,
                mis_rec_threshold=args.mis_rec_threshold,
                stage=1, use_uncertainty_net=False)

            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)

            # ood_pred = torch.sigmoid(ood_logits)

            # pred_label = torch.where(torch.Tensor(pred.detach().cpu()).view(-1, 1) >= args.mis_rec_label_threshold, 1.0, 0.0)
            # ood_label = torch.where(pred_label == y, 1.0, 0.0)

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            # loss_ood = criterion(ood_pred, ood_label.view(-1, 1).to(device))

            auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            # ood_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred.detach().cpu()), np.array(ood_label))

            # optimizer.zero_grad()
            optimizer_uncertainty.zero_grad()
            # loss_total = loss + 0.1 * loss_ood
            loss_total = loss
            loss_total.backward()
            # optimizer.step()

            if fix_layer:
                for name, parms in model_uncertainty_obj.named_parameters():
                    if name not in args.stage1_layer_list:
                        parms.grad = zero_grad_dict[name]

            optimizer_uncertainty.step()

            # if step % log_every == 0:
            #     logger.info(
            #         "epoch={}, step={}, loss={:5f}, ood_loss={:5f}, auc={:5f}, ood_auc={:5f}, speed={:2f} steps/s".format(
            #             epoch, step, float(loss.item()), float(loss_ood.item()), auc, ood_auc, log_every / timer.tick(False)
            #         )
            #     )

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict_stage1(
            predict_dataset=pred_dataloader,
            # model_obj=model_obj,
            model_obj=model_uncertainty_obj,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
            epoch=epoch,
            mis_rec_threshold=args.mis_rec_threshold
        )
        # train_threshold = _threshold
        # train_ood_threshold = _ood_threshold

        logger.info("dump checkpoint for epoch {}".format(epoch))
        # model_obj.train()
        model_uncertainty_obj.train()

        # if pred_ood_auc > best_ood_auc:
        #     best_ood_auc = pred_ood_auc
        #     # torch.save(model_obj, best_ckpt_path)
        #     torch.save(model_uncertainty_obj, best_ckpt_path)

        if pred_overall_auc > best_auc:
            best_auc = pred_overall_auc
            # torch.save(model_obj, best_auc_ckpt_path)
            best_model = deepcopy(model_uncertainty_obj)
            torch.save(model_uncertainty_obj, best_auc_ckpt_path)

    return best_model


def train_stage2(positive_train_dataset, train_dataset, model_uncertainty_obj, model_obj, device, args,
                 positive_pred_dataloader, pred_dataloader, model_load_name_set=None, best_model=None):
    if best_model is not None:
        # model_uncertainty_obj = deepcopy(best_model)
        model_uncertainty_obj = best_model
    optimizer_uncertainty = torch.optim.Adam(
        model_uncertainty_obj.parameters(),
        # lr=float(args.learning_rate)
        lr=float(args.learning_rate_uncertainty)
    )
    # optimizer = torch.optim.Adam(
    #     model_obj.parameters(),
    #     lr=float(args.learning_rate)
    # )
    criterion = nn.BCEWithLogitsLoss()
    model_uncertainty_obj.train()
    model_uncertainty_obj.to(device)
    print(model_uncertainty_obj)

    # model_obj.train()
    # model_obj.to(device)
    # print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200

    max_step = 0

    best_mse = 1e5
    best_auc = 0
    best_mse_ckpt_path = os.path.join(args.checkpoint_dir, "stage2_best_mse" + "_ood.pkl")
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "stage2_best_auc" + ".pkl")

    # train_ood_threshold = 0.2

    # stage2_layer_list = []
    fix_layer = args.stage2_fix_layer
    if fix_layer:
        zero_grad_dict = {}
        for name, parms in model_uncertainty_obj.named_parameters():
            # if name in args.stage1_layer_list:
            if name.split("_")[0] == "stage2":
                args.stage2_layer_list.append(name)
                # print("-" * 50)
                # print(name)
                # print(parms)
                parms.requires_grad = True
            else:
                parms.requires_grad = False
                zero_grad_dict[name] = torch.zeros_like(parms).to(device)

    print("=" * 50)
    print("args.stage1_layer_list ", args.stage1_layer_list)
    print("-" * 50)
    print("args.stage2_layer_list ", args.stage2_layer_list)
    ### stage2: Next-item Prediction model
    # for stage in range(2):
    for epoch in range(1, args.max_epoch_uncertainty + 1):
        torch.cuda.empty_cache()
        for step, batch_data in enumerate(positive_train_dataset, 1):
            logits, loss_cvae = model_uncertainty_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID}
            }, uncertainty_threshold=args.uncertainty_threshold,
                mis_rec_threshold=args.mis_rec_threshold,
                stage=2, use_uncertainty_net=False)

            # loss = loss_cvae
            #
            # optimizer_uncertainty.zero_grad()
            # loss.backward()

            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
            # print("+" * 50)
            # print(pred.size(), y.size())
            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            # print(loss.size())
            # print(loss_cvae.size())
            auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer_uncertainty.zero_grad()
            # loss_total = loss
            # print("-" * 50)
            # print(loss)
            # print(loss_cvae)
            # loss_total = loss + loss_cvae
            # loss_total = -loss + loss_cvae
            # loss_total = loss - loss_cvae
            # loss_total = loss + loss_cvae
            loss_total = loss + abs(loss_cvae)
            loss_total.backward()

            if fix_layer:
                for name, parms in model_uncertainty_obj.named_parameters():
                    if name not in args.stage2_layer_list:
                        parms.grad = zero_grad_dict[name]

            optimizer_uncertainty.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), log_every / timer.tick(False)
                    )
                )

        # pred_mse, pred_uncertainty, _ood_threshold = predict_stage2(
        #     predict_dataset=positive_pred_dataloader,
        #     model_obj=model_uncertainty_obj,
        #     device=device,
        #     args=args,
        #     # bucket=bucket,
        #     train_epoch=epoch,
        #     train_step=epoch * max_step,
        #     # writer=writer
        #     epoch=epoch,
        #     mis_rec_threshold=args.mis_rec_threshold
        # )
        # model_uncertainty_obj.train()
        # if best_mse > pred_mse:
        #     best_mse = pred_mse
        #     best_model = deepcopy(model_uncertainty_obj)
        #     torch.save(model_uncertainty_obj, best_mse_ckpt_path)

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, \
        pred_mse, pred_uncertainty, _ood_threshold = predict_stage2(
            predict_dataset=pred_dataloader,
            # model_obj=model_obj,
            model_obj=model_uncertainty_obj,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
            epoch=epoch,
            mis_rec_threshold=args.mis_rec_threshold
        )
        # train_threshold = _threshold
        # train_ood_threshold = _ood_threshold

        logger.info("dump checkpoint for epoch {}".format(epoch))
        # model_obj.train()
        model_uncertainty_obj.train()

        # if pred_ood_auc > best_ood_auc:
        #     best_ood_auc = pred_ood_auc
        #     # torch.save(model_obj, best_ckpt_path)
        #     torch.save(model_uncertainty_obj, best_ckpt_path)

        if pred_overall_auc > best_auc:
            best_auc = pred_overall_auc
            # torch.save(model_obj, best_auc_ckpt_path)
            best_model = deepcopy(model_uncertainty_obj)
            torch.save(model_uncertainty_obj, best_auc_ckpt_path)

    return best_model


def train_stage3(positive_train_dataset, train_dataset, model_uncertainty_obj, model_obj, device, args,
                 positive_pred_dataloader, pred_dataloader, model_load_name_set=None, best_model=None):
    if best_model is not None:
        # model_uncertainty_obj = deepcopy(best_model)
        model_uncertainty_obj = best_model
    criterion = nn.BCEWithLogitsLoss()
    optimizer_uncertainty = torch.optim.Adam(
        model_uncertainty_obj.parameters(),
        lr=float(args.learning_rate)
    )
    # optimizer = torch.optim.Adam(
    #     model_obj.parameters(),
    #     lr=float(args.learning_rate)
    # )

    model_uncertainty_obj.train()
    model_uncertainty_obj.to(device)
    print(model_uncertainty_obj)

    # model_obj.train()
    # model_obj.to(device)
    # print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200

    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "stage3_best_auc" + ".pkl")

    best_ood_auc = 0

    best_ckpt_path = os.path.join(args.checkpoint_dir, "stage3_best_auc" + "_ood.pkl")

    # train_threshold = 0.5
    # train_ood_threshold = 0.2

    # stage1_layer_list = []
    # for name, parms in model_uncertainty_obj.named_parameters():
    #     args.stage1_layer_list.append(name)
    fix_layer = args.stage3_fix_layer
    if fix_layer:
        zero_grad_dict = {}
        for name, parms in model_uncertainty_obj.named_parameters():
            # if name in args.stage1_layer_list or name in args.stage2_layer_list:
            if name.split("_")[0] == "stage3":
                args.stage3_layer_list.append(name)
                # print("-" * 50)
                # print(name)
                # print(parms)
                parms.requires_grad = True
            else:
                parms.requires_grad = False
                zero_grad_dict[name] = torch.zeros_like(parms).to(device)
        print("args.stage3_layer_list ", args.stage3_layer_list)

    if args.use_uncertainty_net:
        ### stage1: RecSys model and Mis-recommendation model
        for epoch in range(1, args.max_epoch + 1):
            ood_pred_list = []
            for step, batch_data in enumerate(train_dataset, 1):
                # logits, ood_logits = model_obj({
                logits, ood_logits = model_uncertainty_obj({
                    key: value.to(device)
                    for key, value in batch_data.items()
                    if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                    # }, train_ood_threshold=train_ood_threshold)
                }, uncertainty_threshold=args.uncertainty_threshold,
                    mis_rec_threshold=args.mis_rec_threshold,
                    stage=3, use_uncertainty_net=True)

                pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)

                ood_pred = torch.sigmoid(ood_logits)
                ood_pred_list.extend(ood_pred.detach().cpu().numpy().tolist())
                pred_label = torch.where(torch.Tensor(pred.detach().cpu()).view(-1, 1) >= args.mis_rec_label_threshold, 1.0, 0.0)
                ood_label = torch.where(pred_label == y, 1.0, 0.0)

                loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
                if args.focal_loss:
                    # loss_ood = criterion(ood_pred, ood_label.view(-1, 1).to(device))
                    import kornia
                    # criterion_focal_loss = kornia.losses.binary_focal_loss_with_logits()
                    # loss_ood = criterion(ood_pred, ood_label.view(-1, 1).to(device))
                    # print("BCE ", loss_ood)
                    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
                    loss_ood = kornia.losses.binary_focal_loss_with_logits(
                        # input=ood_pred, target=ood_label.to(device), **kwargs
                        input=ood_logits, target=ood_label.to(device), **kwargs
                    ) * 10
                    # print("Focal BCE ", loss_ood)
                else:
                    # loss_ood = criterion(ood_pred, ood_label.view(-1, 1).to(device))
                    loss_ood = criterion(ood_logits, ood_label.view(-1, 1).to(device))

                auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                ood_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred.detach().cpu()), np.array(ood_label))

                # optimizer.zero_grad()
                optimizer_uncertainty.zero_grad()
                if fix_layer:
                    loss_total = loss_ood
                else:
                    loss_total = loss + 0.1 * loss_ood

                loss_total.backward()
                # optimizer.step()

                if fix_layer:
                    for name, parms in model_uncertainty_obj.named_parameters():
                        if name not in args.stage3_layer_list:
                            parms.grad = zero_grad_dict[name]

                optimizer_uncertainty.step()

                if step % log_every == 0:
                    logger.info(
                        "epoch={}, step={}, loss={:5f}, ood_loss={:5f}, auc={:5f}, ood_auc={:5f}, speed={:2f} steps/s".format(
                            epoch, step, float(loss.item()), float(loss_ood.item()), auc, ood_auc, log_every / timer.tick(False)
                        )
                    )
                max_step = step
            # args.uncertainty_threshold = torch.Tensor(sorted(ood_pred_list, reverse=True)[len(ood_pred_list) * 10 // 100])
            # args.uncertainty_threshold = torch.Tensor(sorted(ood_pred_list)[len(ood_pred_list) * 10 // 100])
            # args.mis_rec_threshold = torch.Tensor(sorted(ood_pred_list, reverse=True)[len(ood_pred_list) * 10 // 100])
            args.mis_rec_threshold = torch.Tensor(sorted(ood_pred_list)[len(ood_pred_list) * 10 // 100])

            pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
            pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, pred_ood_auc, \
            _threshold, _ood_threshold = predict_stage3(
                predict_dataset=pred_dataloader,
                # model_obj=model_obj,
                model_obj=model_uncertainty_obj,
                device=device,
                args=args,
                # bucket=bucket,
                train_epoch=epoch,
                train_step=epoch * max_step,
                # writer=writer
                epoch=epoch,
                mis_rec_threshold=args.mis_rec_threshold
            )
            train_threshold = _threshold
            train_ood_threshold = _ood_threshold

            logger.info("dump checkpoint for epoch {}".format(epoch))
            # model_obj.train()
            model_uncertainty_obj.train()

            if pred_ood_auc > best_ood_auc:
                best_ood_auc = pred_ood_auc
                # torch.save(model_obj, best_ckpt_path)
                torch.save(model_uncertainty_obj, best_ckpt_path)

            if pred_overall_auc > best_auc:
                best_auc = pred_overall_auc
                # torch.save(model_obj, best_auc_ckpt_path)
                best_model = deepcopy(model_uncertainty_obj)
                torch.save(model_uncertainty_obj, best_auc_ckpt_path)
    else:
        epoch = 0
        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, pred_ood_auc, \
        _threshold, _ood_threshold = predict_stage3(
            predict_dataset=pred_dataloader,
            # model_obj=model_obj,
            model_obj=model_uncertainty_obj,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
            epoch=epoch,
            mis_rec_threshold=args.mis_rec_threshold
        )

    return best_model


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
    return model_obj, model_load_name_set


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    # bucket = oss_io.open_bucket(args.bucket)

    # Check if the specified path has an existed model
    # if bucket.object_exists(args.checkpoint_dir):
    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, "meta")
    # if os.path.exists(args.checkpoint_dir):
    #     raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)
    # else:
    #     os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_uncertainty = model.get_model_meta(args.model_uncertainty)  # type: model.ModelMeta
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict

    model_uncertainty_conf, raw_model_uncertainty_conf = ap.parse_arch_config_from_args(model_uncertainty, args)  # type: dict
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
    model_uncertainty_obj = model_uncertainty.model_builder(model_conf=model_uncertainty_conf)  # type: torch.nn.module
    model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
    # model_obj, model_load_name_set = load_model(args, model_obj)

    # print("=" * 100)
    # for name, parms in model_obj.named_parameters():
    #     print(name)
    # print("=" * 100)
    device = env.get_device()
    # worker_id, worker_count = env.get_cluster_info()
    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(',')
    positive_train_file, positive_test_file = args.positive_dataset.split(',')

    args.stage1_layer_list = []
    args.stage2_layer_list = []
    args.stage3_layer_list = []

    args.uncertainty_threshold = 0.3
    args.mis_rec_threshold = 0.9
    args.mis_rec_label_threshold = 0.5

    best_model = None

    # args.stage1_fix_layer = False
    # args.stage2_fix_layer = False
    # args.stage3_fix_layer = False

    args.stage1_fix_layer = True
    args.stage2_fix_layer = True
    args.stage3_fix_layer = True

    # args.use_uncertainty_net = False
    args.use_uncertainty_net = True

    # args.focal_loss = True
    args.focal_loss = False

    # important!!!!
    args.num_loading_workers = 1
    # Setup up data loader
    train_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_file,
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_dataloader.batchify,
        drop_last=False,
        # shuffle=True
    )

    # Setup up data loader
    pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataloader,
        batch_size=args.batch_size,
        # batch_size=1,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False
    )

    # Setup up data loader
    positive_train_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        # table_name=positive_train_file,
        table_name=train_file,
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    positive_train_dataloader = torch.utils.data.DataLoader(
        positive_train_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=positive_train_dataloader.batchify,
        drop_last=False,
        # shuffle=True
    )

    # Setup up data loader
    positive_pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        # table_name=positive_test_file,
        table_name=test_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    positive_pred_dataloader = torch.utils.data.DataLoader(
        positive_pred_dataloader,
        batch_size=args.batch_size,
        # batch_size=1,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=positive_pred_dataloader.batchify,
        drop_last=False
    )

    # # Setup training
    best_model = train_stage1(
        positive_train_dataset=positive_train_dataloader,
        train_dataset=train_dataloader,
        model_uncertainty_obj=model_uncertainty_obj,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        positive_pred_dataloader=positive_pred_dataloader,
        pred_dataloader=pred_dataloader,
        # model_load_name_set=model_load_name_set,
        best_model=best_model
    )

    best_model = train_stage2(
        positive_train_dataset=positive_train_dataloader,
        train_dataset=train_dataloader,
        model_uncertainty_obj=model_uncertainty_obj,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        positive_pred_dataloader=positive_pred_dataloader,
        pred_dataloader=pred_dataloader,
        # model_load_name_set=model_load_name_set,
        best_model=best_model
    )

    best_model = train_stage3(
        positive_train_dataset=positive_train_dataloader,
        train_dataset=train_dataloader,
        model_uncertainty_obj=model_uncertainty_obj,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        positive_pred_dataloader=positive_pred_dataloader,
        pred_dataloader=pred_dataloader,
        # model_load_name_set=model_load_name_set,
        best_model=best_model
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

