# coding=utf-8
import os
import logging
import argparse
import sys

import common_io.table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp

import model
from util.timer import Timer
from util import oss_io
from util import args_processing as ap
from util import consts
from util import path
from util import env
from loader import sequence_dataloader
# import ipdb
from sklearn import metrics
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--outputs", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--step", type=int, help="Number of iterations to dump model")
    parser.add_argument("--batch_size", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--num_loading_workers", type=int, default=8)

    return parser.parse_known_args()[0]


# def predict(predict_dataset, model_obj, device, args, bucket, writer):
def predict(predict_dataset, model_obj, device, args, writer, feature_file):
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

    # timer = Timer()
    # log_every = 200
    # WRITING_BATCH_SIZE = 512

    buffer = []
    trigger_tensor_list = []
    y_list = []
    prob_list = []
    # model_obj.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(predict_dataset, 1):
            logits, trigger_tensor = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            })
            trigger_tensor_list.append(trigger_tensor)
            prob = torch.sigmoid(logits)
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)
            y_list.extend(y)
            prob_list.extend(prob.detach().cpu())
            # fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(prob.detach().cpu()), pos_label=1)
            # auc = float(metrics.auc(fpr, tpr))
            buffer.extend(
                [str(user_id), float(score), float(label)]
                for user_id, score, label
                in zip(
                    batch_data[consts.FIELD_USER_ID],
                    batch_data[consts.FIELD_LABEL],
                    prob.detach().cpu().numpy()
                )
            )

        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_list), np.array(prob_list), pos_label=1)
        auc = float(metrics.auc(fpr, tpr))
        print("test auc: ", auc)
        print("test auc: ", auc, file=writer)
        torch.save(trigger_tensor_list, feature_file_path)

        #     if step % log_every == 0:
        #         logger.info(
        #             "step={}, auc={:5f}, speed={:2f} steps/s".format(
        #                 step, auc, log_every / timer.tick(False)
        #             )
        #         )
        #
        #     if len(buffer) >= WRITING_BATCH_SIZE:
        #         writer.write(buffer, [0, 1, 2])
        #         buffer = []
        #
        # if len(buffer) >= 0:
        #     writer.write(buffer, [0, 1, 2])


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    # bucket = oss_io.open_bucket(args.bucket)

    # print("=" * 50)
    # print("train_args")
    # print(ipdb.trace())
    # print("=" * 50)
    # Check if the specified path has an existed model
    # train_args = ap.load_train_arguments(args.checkpoint_dir, bucket)

    # model_meta = model.get_model_meta(train_args.model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf = ap.load_arch_config(model_meta, args.checkpoint_dir, bucket)

    # Construct model
    # model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module

    device = env.get_device()
    # worker_id, worker_count = env.get_cluster_info()
    ckpt_path = os.path.join(args.checkpoint_dir, "best.pkl")
    output_file = os.path.join(args.checkpoint_dir, "test.txt")
    feature_file = os.path.join(args.checkpoint_dir, "feature.pt")

    model_obj = torch.load(ckpt_path)

    # Setup up data loader
    pred_dataloader = sequence_dataloader.SequenceDataLoader(
        # table_name=args.tables.split(',')[0],
        table_name=args.dataset,
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
    # with common_io.table.TableWriter(args.outputs, slice_id=worker_id) as writer:
    with open(output_file, "w+") as writer:
        predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            # bucket=bucket,
            writer=writer,
            feature_file=feature_file
        )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)
