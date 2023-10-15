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
from util import metrics
# from loader import sequence_dataloader
# from loader import new_meta_sequence_dataloader as meta_sequence_dataloader
from loader import meta_ood_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile
from tqdm import tqdm

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


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0):
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
    ood_pred_list = []
    ood_y_list = []

    buffer = []
    _request_num = 0
    _total_num = 0
    for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
        logits, ood_logits, request_num, total_num = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        }, pred=True)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)

        ood_prob = torch.sigmoid(ood_logits).detach().cpu().numpy()
        ood_y = torch.where(torch.argmax(torch.Tensor(prob), dim=1).view(-1, 1) == y, 1.0, 0.0)

        # auc = metrics.calculate_auc(prob, y)
        # ood_auc = metrics.calculate_auc(ood_prob, np.array(ood_y))

        # ndcg = metrics.calculate_ndcg(10, prob, y)
        ndcg = 0

        _request_num += request_num
        _total_num += total_num

        pred_list.extend(prob)
        y_list.extend(np.array(y))
        ood_pred_list.extend(ood_prob)
        ood_y_list.extend(np.array(ood_y))
        # if epoch >= 5:
        #     print("=" * 50)
        #     print(ood_pred_list)
        #     print("-" * 50)
        #     print(ood_y_list)
        buffer.extend(
            [str(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                batch_data[consts.FIELD_LABEL],
                prob
            )
        )

        # if step % log_every == 0:
        #     logger.info(
        #         "train_epoch={}, step={}, auc={:5f}, ood_auc={:5f}, ndcg={:5f}, request_num={}, total_num={}, speed={:2f} steps/s".format(
        #             train_epoch, step, auc, ood_auc, ndcg, request_num, total_num, log_every / timer.tick(False)
        #         )
        #     )

        # if len(buffer) >= WRITING_BATCH_SIZE:
        #     writer.write(buffer, [0, 1, 2])
        #     buffer = []

    # if len(buffer) >= 0:
    #     writer.write(buffer, [0, 1, 2])

    auc = metrics.calculate_auc(np.array(pred_list), np.array(y_list))
    ood_auc = metrics.calculate_auc(np.array(ood_pred_list), np.array(ood_y_list))
    # ndcg = metrics.calculate_ndcg(10, np.array(pred_list), np.array(y_list))
    ndcg = 0

    print("train_epoch={}, train_step={}, auc={:5f}, ood_auc={:5f}, ndcg={:5f}, request_num={}, total_num={}".format(
        train_epoch, train_step, auc, ood_auc, ndcg, _request_num, _total_num
    ))
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, auc={:5f}, ood_auc={:5f}, ndcg={:5f}, request_num={}, total_num={}".format(
            train_epoch, train_step, auc, ood_auc, ndcg, _request_num, _total_num
        ), file=writer)

    return auc, ood_auc


# def train(train_dataset, model_obj, device, args, bucket, pred_dataloader):
def train(train_dataset, model_obj, device, args, pred_dataloader, model_load_name_set=None):
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
    best_ood_auc = 0
    # best_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + "_ood.pkl")
    # zero_grad_dict = {}
    # for name, parms in model_obj.named_parameters():
    #     zero_grad_dict[name] = torch.zeros_like(parms).to(device)
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            logits, ood_logits = model_obj({
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

            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)

            ood_pred = torch.sigmoid(ood_logits)
            ood_label = torch.where(torch.argmax(torch.Tensor(pred.detach().cpu()), dim=1).view(-1, 1) == y, 1.0, 0.0)

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            loss_ood = criterion(ood_pred, ood_label.view(-1, 1).to(device))

            # fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred.detach().cpu()), pos_label=1)
            # auc = float(metrics.auc(fpr, tpr))
            # print("np.array(pred.detach().cpu()) ", np.array(pred.detach().cpu()))
            # print("np.array(y) ", np.array(y))
            # print("np.array(pred.detach().cpu()).shape ", np.array(pred.detach().cpu()).shape)
            # print("np.array(y).shape ", np.array(y).shape)
            # auc = metrics.calculate_auc(np.array(pred.detach().cpu()), np.array(y.int()))

            auc = metrics.calculate_auc(np.array(pred.detach().cpu()), np.array(y))
            ood_auc = metrics.calculate_auc(np.array(ood_pred.detach().cpu()), np.array(ood_label))
            optimizer.zero_grad()
            loss_total = loss + 0.2 * loss_ood
            # loss_ood.backward()
            # loss.backward()
            loss_total.backward()

            # for name, parms in model_obj.named_parameters():
            #     if name in model_load_name_set:
            #         # print("-" * 50)
            #         # print(name)
            #         # parms.grad = torch.zeros_like(parms)
            #         # print(parms.grad)
            #         parms.grad = zero_grad_dict[name]
            #         # print(parms.grad)

            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, ood_loss={:5f}, auc={:5f}, ood_auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), float(loss_ood.item()), auc, ood_auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_auc, pred_ood_auc = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
            epoch=epoch
        )
        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + str(step) + ".pkl")

        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + "epoch_" + str(epoch) + "_ood.pkl")
        # torch.save(model_obj, ckpt_path)
        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj.train()
        # if pred_auc > best_auc:
        #     best_auc = pred_auc
        #     torch.save(model_obj, best_ckpt_path)
        if pred_ood_auc > best_ood_auc:
            best_ood_auc = pred_ood_auc
            torch.save(model_obj, best_ckpt_path)


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
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
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

    # Setup training
    train(
        train_dataset=train_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        pred_dataloader=pred_dataloader,
        # model_load_name_set=model_load_name_set
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

