import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

ROOT_FOLDER1 = "../checkpoint/SIGIR2023"
# ROOT_FOLDER2 = "../checkpoint/SIGIR2023_backup"
ROOT_FOLDER2 = "../checkpoint/SIGIR2023"

# DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_clothing", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty", "movielens", ]

MODEL_LIST = ["din", "sasrec", "gru4rec"]

# TYPE_LIST = ["base", "base_finetune", "meta", "meta_gru", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_if", "meta_ood", "meta_ood_gru", "meta_ood_uncertainty"]
TYPE_LIST1 = ["base", "base_finetune", "meta", "meta_ood_ocsvm", "meta_ood_lof"]
TYPE_LIST2 = ["meta_random", "meta_ood", "meta_ood_uncertainty"]

log_filename = "log_ood.txt"

txt_file = os.path.join(ROOT_FOLDER2, "result_ood_merge.txt")
csv_file = os.path.join(ROOT_FOLDER2, "result_ood_merge.csv")

epoch = 10


def print_dataset(dataset, writer):
    print("=" * 50, file=writer)
    print(dataset, file=writer)
    print("-" * 50, file=writer)


with open(txt_file, "w+") as txt_writer, open(csv_file, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print_dataset(dataset, txt_writer)
        print_dataset(dataset, csv_writer)

        for model in MODEL_LIST:
            for type in TYPE_LIST1:
                # init_list()
                # auc_list = auc_user_list = logloss_list = ndcg5_list = ndcg10_list = ndcg20_list = hr5_list = hr10_list = hr20_list = request_num_list = total_num_list = []
                auc_list = []
                auc_user_list = []
                logloss_list = []
                ndcg5_list = []
                ndcg10_list = []
                ndcg20_list = []
                hr5_list = []
                hr10_list = []
                hr20_list = []
                rate_list = []

                # auc_max_list = auc_user_max_list = logloss_max_list = ndcg5_max_list = ndcg10_max_list = ndcg20_max_list = hr5_max_list = hr10_max_list = hr20_max_list = request_num_max_list = total_num_max_list = []
                # auc_max_list = auc_user_max_list = logloss_max_list = ndcg5_max_list = ndcg10_max_list = ndcg20_max_list = hr5_max_list = hr10_max_list = hr20_max_list = request_num_max_list = total_num_max_list = []
                auc_max_list = []
                auc_user_max_list = []
                logloss_max_list = []
                ndcg5_max_list = []
                ndcg10_max_list = []
                ndcg20_max_list = []
                hr5_max_list = []
                hr10_max_list = []
                hr20_max_list = []
                request_num_list = []
                total_num_list = []
                rate_max_list = []
                # max_auc = max_auc_user = max_logloss = max_ndcg5 = max_ndcg10 = max_ndcg20 = max_hr5 = max_hr10 = max_hr20 = 0

                log_file = os.path.join(ROOT_FOLDER2, "{}_{}".format(dataset, model), type, log_filename)

                if type == "base_finetune":
                    log_file = os.path.join(ROOT_FOLDER2, "{}_{}".format(dataset, model), type, "log_overall.txt")
                if not os.path.exists(log_file):
                    # print("{} is not found".format(log_file))
                    continue

                with open(log_file, "r+") as reader:
                    for index, line in enumerate(reader, 1):
                        # print(line.strip("\n"))
                        if index > 50:
                            break

                        epoch_num = int(line.strip("\n").split(",")[0].split("=")[-1])
                        auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                        auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                        logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                        ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                        hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                        ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                        hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                        ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                        hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])

                        if type == "meta_ood_ocsvm" or type == "meta_ood_lof":
                            request_num = int(line.strip("\n").split(",")[-4].split("=")[-1])
                            total_num = int(line.strip("\n").split(",")[-3].split("=")[-1])
                            # print("=" * 30)
                            # print(request_num)
                            # print(total_num)
                            request_num_list.append(request_num)
                            total_num_list.append(total_num)

                        auc_list.append(auc)
                        auc_user_list.append(auc_user)
                        logloss_list.append(logloss)
                        ndcg5_list.append(ndcg5)
                        ndcg10_list.append(ndcg10)
                        ndcg20_list.append(ndcg20)
                        hr5_list.append(hr5)
                        hr10_list.append(hr10)
                        hr20_list.append(hr20)

                        if type == "base_finetune":
                            auc_max_list.append(auc)
                            auc_user_max_list.append(auc_user)
                            logloss_max_list.append(logloss)
                            ndcg5_max_list.append(ndcg5)
                            ndcg10_max_list.append(ndcg10)
                            ndcg20_max_list.append(ndcg20)
                            hr5_max_list.append(hr5)
                            hr10_max_list.append(hr10)
                            hr20_max_list.append(hr20)
                            # break
                            continue

                        if index % epoch == 0:
                            # print(type)
                            auc_max_list.append(max(auc_list))
                            auc_user_max_list.append(max(auc_user_list))
                            try:
                                logloss_max_list.append(min(logloss_list))
                            except ValueError:
                                logloss_max_list.append(np.nan)
                            ndcg5_max_list.append(max(ndcg5_list))
                            ndcg10_max_list.append(max(ndcg10_list))
                            ndcg20_max_list.append(max(ndcg20_list))
                            hr5_max_list.append(max(hr5_list))
                            hr10_max_list.append(max(hr10_list))
                            hr20_max_list.append(max(hr20_list))
                            # request_num_max_list.append(request_num_list[auc_user_list.index(max(auc_user_list))])
                            # total_num_max_list.append(total_num_list[auc_user_list.index(max(auc_user_list))])
                            if type == "base" or type == "base_finetune":
                                # print(1)
                                rate_max_list.append(0)
                            elif type == "meta":
                                # print(2)
                                rate_max_list.append(1)
                            elif type == "meta_ood_ocsvm" or type == "meta_ood_lof":
                                # print(3)
                                # print("-" * 30)
                                # print(request_num_list[auc_user_list.index(max(auc_user_list))])
                                # print(total_num_list[auc_user_list.index(max(auc_user_list))])
                                # print(request_num_list[auc_user_list.index(max(auc_user_list))]
                                #                      /
                                #                      total_num_list[auc_user_list.index(max(auc_user_list))])
                                rate_max_list.append(request_num_list[auc_user_list.index(max(auc_user_list))]
                                                     /
                                                     total_num_list[auc_user_list.index(max(auc_user_list))])

                            # auc_list = auc_user_list = logloss_list = ndcg5_list = ndcg10_list = ndcg20_list = hr5_list = hr10_list = hr20_list = request_num_list = total_num_list = []
                            auc_list = []
                            auc_user_list = []
                            logloss_list = []
                            ndcg5_list = []
                            ndcg10_list = []
                            ndcg20_list = []
                            hr5_list = []
                            hr10_list = []
                            hr20_list = []
                            request_num_list = []
                            total_num_list = []
                            rate_list = []

                    if len(auc_list) != 0:
                        auc_max_list.append(max(auc_list))
                        auc_user_max_list.append(max(auc_user_list))
                        try:
                            logloss_max_list.append(min(logloss_list))
                        except ValueError:
                            logloss_max_list.append(np.nan)
                        ndcg5_max_list.append(max(ndcg5_list))
                        ndcg10_max_list.append(max(ndcg10_list))
                        ndcg20_max_list.append(max(ndcg20_list))
                        hr5_max_list.append(max(hr5_list))
                        hr10_max_list.append(max(hr10_list))
                        hr20_max_list.append(max(hr20_list))
                        # request_num_max_list.append(request_num_list[auc_user_list.index(max(auc_user_list))])
                        # total_num_max_list.append(total_num_list[auc_user_list.index(max(auc_user_list))])
                        if type == "base" or "base_finetune":
                            rate_max_list.append(0)
                        elif type == "meta":
                            rate_max_list.append(1)
                        elif type == "meta_ood_ocsvm" or type == "meta_ood_lof":
                            rate_max_list.append(request_num_list[auc_user_list.index(max(auc_user_list))]
                                                 /
                                                 total_num_list[auc_user_list.index(max(auc_user_list))])
                        # auc_list = auc_user_list = logloss_list = ndcg5_list = ndcg10_list = ndcg20_list = hr5_list = hr10_list = hr20_list = request_num_list = total_num_list = []
                        auc_list = []
                        auc_user_list = []
                        logloss_list = []
                        ndcg5_list = []
                        ndcg10_list = []
                        ndcg20_list = []
                        hr5_list = []
                        hr10_list = []
                        hr20_list = []
                        request_num_list = []
                        total_num_list = []
                        rate_list = []

                print(
                    model, type,
                    round(float(np.average(auc_max_list)), 4), round(float(np.std(auc_max_list)), 4),
                    round(float(np.average(auc_user_max_list)), 4), round(float(np.std(auc_user_max_list)), 4),
                    # round(float(np.average(logloss_max_list)), 4), round(float(np.std(logloss_max_list)), 4),
                    round(float(np.average(ndcg5_max_list)), 4), round(float(np.std(ndcg5_max_list)), 4),
                    round(float(np.average(hr5_max_list)), 4), round(float(np.std(hr5_max_list)), 4),
                    round(float(np.average(ndcg10_max_list)), 4), round(float(np.std(ndcg10_max_list)), 4),
                    round(float(np.average(hr10_max_list)), 4), round(float(np.std(hr10_max_list)), 4),
                    round(float(np.average(ndcg20_max_list)), 4), round(float(np.std(ndcg20_max_list)), 4),
                    round(float(np.average(hr20_max_list)), 4), round(float(np.std(hr20_max_list)), 4),
                    # round(float(np.average(request_num_max_list)) / float(np.average(total_num_max_list)), 4),
                    round(float(np.average(rate_max_list)), 4), sep=",", file=csv_writer)
                
            for type in TYPE_LIST2:
                # log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                log_file = os.path.join(ROOT_FOLDER1, "{}_{}".format(dataset, model), type, "test.txt")
                auc_dict = defaultdict(list)
                auc_user_dict = defaultdict(list)
                logloss_dict = defaultdict(list)
                ndcg5_dict = defaultdict(list)
                ndcg10_dict = defaultdict(list)
                ndcg20_dict = defaultdict(list)
                hr5_dict = defaultdict(list)
                hr10_dict = defaultdict(list)
                hr20_dict = defaultdict(list)
                request_num_dict = defaultdict(list)
                total_num_dict = defaultdict(list)
                rate_dict = defaultdict(list)
                if not os.path.exists(log_file):
                    print(log_file)
                with open(log_file, "r+") as reader:
                    for index, line in enumerate(reader, 1):
                        epoch_num = int(line.strip("\n").split(",")[0].split("=")[-1])
                        auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                        auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                        # logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                        ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                        hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                        ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                        hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                        ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                        hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])
                        rate = float(line.strip("\n").split(",")[-1].split("=")[-1])

                        auc_dict[rate].append(auc)
                        auc_user_dict[rate].append(auc_user)
                        # logloss_dict[rate].append(logloss)
                        ndcg5_dict[rate].append(ndcg5)
                        ndcg10_dict[rate].append(ndcg10)
                        ndcg20_dict[rate].append(ndcg20)
                        hr5_dict[rate].append(hr5)
                        hr10_dict[rate].append(hr10)
                        hr20_dict[rate].append(hr20)

                        # request_num_dict[rate].append()
                        # total_num_dict[rate].append()

                        # auc_list.append(auc)
                        # auc_user_list.append(auc_user)
                        # logloss_list.append(logloss)
                        # ndcg5_list.append(ndcg5)
                        # ndcg10_list.append(ndcg10)
                        # ndcg20_list.append(ndcg20)
                        # hr5_list.append(hr5)
                        # hr10_list.append(hr10)
                        # hr20_list.append(hr20)
                        # rate_list.append(rate)
                        if rate == float(100):
                            break
                    for r in [0.1, 1, 10]:
                        print(
                            model, type,
                            round(float(np.average(auc_dict[r])), 4), round(float(np.std(auc_dict[r])), 4),
                            round(float(np.average(auc_user_dict[r])), 4), round(float(np.std(auc_user_dict[r])), 4),
                            # round(float(np.average(logloss_dict[r])), 4), round(float(np.std(logloss_dict[r])), 4),
                            round(float(np.average(ndcg5_dict[r])), 4), round(float(np.std(ndcg5_dict[r])), 4),
                            round(float(np.average(hr5_dict[r])), 4), round(float(np.std(hr5_dict[r])), 4),
                            round(float(np.average(ndcg10_dict[r])), 4), round(float(np.std(ndcg10_dict[r])), 4),
                            round(float(np.average(hr10_dict[r])), 4), round(float(np.std(hr10_dict[r])), 4),
                            round(float(np.average(ndcg20_dict[r])), 4), round(float(np.std(ndcg20_dict[r])), 4),
                            round(float(np.average(hr20_dict[r])), 4), round(float(np.std(hr20_dict[r])), 4),
                            # round(float(np.average(request_num_dict[r])) / float(np.average(total_num_dict[r])), 4),
                            float(r / 100), sep=",", file=csv_writer)