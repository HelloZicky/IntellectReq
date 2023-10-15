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
# from loader import sequence_dataloader
# from loader import new_meta_sequence_dataloader2 as meta_sequence_dataloader
from loader import multi_metric_meta_ood_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile

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

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    # parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch", type=int,  default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None):
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
    y_list = []
    buffer = []
    user_id_list = []
    _request_num = 0
    _total_num = 0
    for step, batch_data in enumerate(predict_dataset, 1):
        # logits, request_num, total_num = model_obj({
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        # }, pred=True)
        }, pred=False)
        request_num = 0
        total_num = 0
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        # ndcg = 0
        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        
        _request_num += request_num
        _total_num += total_num

        pred_list.extend(prob)
        y_list.extend(np.array(y))


        # buffer.extend(
        #     [str(user_id), float(score), float(label)]
        #     for user_id, score, label
        #     in zip(
        #         batch_data[consts.FIELD_USER_ID],
        #         batch_data[consts.FIELD_LABEL],
        #         prob
        #     )
        # )
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

        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                    train_epoch, step, overall_auc, log_every / timer.tick(False)
                )
            )

        # if len(buffer) >= WRITING_BATCH_SIZE:
        #     writer.write(buffer, [0, 1, 2])
        #     buffer = []

    # if len(buffer) >= 0:
    #     writer.write(buffer, [0, 1, 2])

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
    # ndcg = 0

    # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
    #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #           format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
    #
    # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
          "request_num={}, total_num={}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
                 _request_num, _total_num))
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print(
            "train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}, "
            "request_num={}, total_num={}".
                format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                       user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20,
                       _request_num, _total_num),
            file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


# def train(train_dataset, model_obj, device, args, bucket, pred_dataloader):
def train(train_dataset, model_obj, device, args, pred_dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj.train()
    model_obj.to(device)

    print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    # writer = open(os.path.join(args.checkpoint_dir, "log.txt"), "a")
    max_step = 0
    best_auc = 0
    best_user_auc = 0
    best_logloss = 0
    best_ndcg = 0
    best_hr = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    best_user_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_user_auc" + ".pkl")
    best_logloss_ckpt_path = os.path.join(args.checkpoint_dir, "best_logloss" + ".pkl")
    best_ndcg_ckpt_path = os.path.join(args.checkpoint_dir, "best_ndcg" + ".pkl")
    best_hr_ckpt_path = os.path.join(args.checkpoint_dir, "best_hr" + ".pkl")
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            })

            # if step == 1:
            #     # from thop import profile
            #     flops, params = profile(model_obj,
            #         {key: value.to(device)
            #          for key, value in batch_data.items()
            #          if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            #          },)
            #     print("flops={}B, params={}M".format(flops / 1e9, params / 1e6))

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
            # fpr, tpr, thresholds = new_metrics.roc_curve(np.array(y), np.array(pred.detach().cpu()), pos_label=1)
            # auc = float(new_metrics.auc(fpr, tpr))
            # print("np.array(pred.detach().cpu()) ", np.array(pred.detach().cpu()))
            # print("np.array(y) ", np.array(y))
            # print("np.array(pred.detach().cpu()).shape ", np.array(pred.detach().cpu()).shape)
            # print("np.array(y).shape ", np.array(y).shape)
            # auc = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y.int()))
            auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
        )
        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + str(step) + ".pkl")

        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + "epoch_" + str(epoch) + ".pkl")
        # torch.save(model_obj, ckpt_path)
        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj.train()
        if pred_overall_auc > best_auc:
            best_auc = pred_overall_auc
            torch.save(model_obj, best_auc_ckpt_path)

        # if pred_user_auc > best_user_auc:
        #     best_user_auc = pred_user_auc
        #     torch.save(model_obj, best_user_auc_ckpt_path)
        #
        # if pred_overall_logloss > best_logloss:
        #     best_logloss = pred_overall_logloss
        #     torch.save(model_obj, best_logloss_ckpt_path)
        #
        # if pred_user_ndcg > best_ndcg:
        #     best_ndcg = pred_user_ndcg
        #     torch.save(model_obj, best_ndcg_ckpt_path)
        #
        # if pred_user_hr > best_hr:
        #     best_hr = pred_user_hr
        #     torch.save(model_obj, best_hr_ckpt_path)


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
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
    model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
    print("=" * 100)
    for name, parms in model_obj.named_parameters():
        print(name)
    print("=" * 100)
    device = env.get_device()
    # worker_id, worker_count = env.get_cluster_info()
    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(',')

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
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False
    )

    # Setup training
    train(
        train_dataset=train_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        pred_dataloader=pred_dataloader
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

