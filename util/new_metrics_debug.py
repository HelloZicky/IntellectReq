from sklearn import metrics
import heapq
import numpy as np
import math
import itertools
import torch


def calculate_user_ndcg_hr(*buffer):
    # buffer.sort()
    user_num = 0
    ndcg_ = 0
    hr_ = 0
    # top_items = heapq.nlargest(3, list(zip(buffer[1], buffer[2])))
    # print(top_items)
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        # actions = sorted(actions, key=lambda x: x[2])
        # top_items = heapq.nlargest(buffer[0], list(zip(buffer[1], buffer[2])))
        # top_items = heapq.nlargest(3, list(zip(actions[1], actions[2])))
        # print(top_items)
        # top_items = heapq.nlargest(5, list(zip(buffer[1], buffer[2])))
        # num_postive = int(sum(buffer[2]))
        actions = sorted(actions, key=lambda x: x[1], reverse=True)
        top_items = np.array(actions)[:3, :]
        num_postive = int(sum(np.array(actions)[:, 2]))
        # print(actions)
        print(top_items)
        dcg = 0
        idcg = 0
        for i, (user_id, score, label) in enumerate(top_items):
            if label == 1:
                dcg += math.log(2) / math.log(i + 2)

            if i < num_postive:
                idcg += math.log(2) / math.log(i + 2)

        ndcg_ += dcg / idcg
        hr_ += 1 if any(item[2] for item in top_items) else 0
        user_num += 1
        print(dcg)
        print(idcg)
        print(ndcg_, hr_)
        print("-" * 50)
    ndcg = ndcg_ / user_num
    hr = hr_ / user_num
    return ndcg, hr


# def calculate_user_ndcg(*buffer):
#     top_items = heapq.nlargest(buffer[0], list(zip(buffer[1], buffer[2])))
#     num_postive = int(sum(buffer[2]))
#
#     dcg = 0
#     idcg = 0
#     for i, (score, label) in enumerate(top_items):
#         if label == 1:
#             dcg += math.log(2) / math.log(i + 2)
#
#         if i < num_postive:
#             idcg += math.log(2) / math.log(i + 2)
#
#     return dcg / idcg


def calculate_overall_auc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), prob, pos_label=1)
    auc = float(metrics.auc(fpr, tpr))

    return auc


if __name__ == '__main__':
    records = [
        [3, 0.151, 0],
        [3, 0.152, 0],
        [3, 0.103, 0],
        [3, 0.174, 1],
        [3, 0.135, 0],
        [3, 0.126, 0],
        [1, 0.151, 1],
        [1, 0.152, 0],
        [1, 0.103, 0],
        [1, 0.174, 0],
        [1, 0.135, 0],
        [2, 0.151, 0],
        [2, 0.152, 0],
        [2, 0.103, 1],
        [2, 0.174, 0],
        [2, 0.135, 0],
        [2, 0.126, 0],
    ]
    # ndcg, hr = calculate_user_ndcg_hr(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2])
    # ndcg, hr = calculate_user_ndcg_hr(np.hstack(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2]))
    # ndcg, hr = calculate_user_ndcg_hr(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1))
    # user_id_list = torch.Tensor([3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,]).view(-1, 1)
    # score_list = torch.Tensor([0.15, 0.15, 0.10, 0.17, 0.13, 0.12, 0.15, 0.15, 0.10, 0.17, 0.13, 0.15, 0.15, 0.10, 0.17, 0.13, 0.12,]).view(-1, 1)
    # label_list = torch.Tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,]).view(-1, 1)
    # print(torch.Tensor(np.array(records)[:, 0]).view(-1, 1))
    # ndcg, hr = calculate_user_ndcg_hr(torch.cat(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1), dim=1))
    # ndcg, hr = calculate_user_ndcg_hr(user_id_list, score_list, label_list)
    ndcg, hr = calculate_user_ndcg_hr(records)
    print(ndcg, hr)
