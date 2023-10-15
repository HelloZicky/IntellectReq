import math
import os
import numpy
import numpy as np
from collections import defaultdict

# ROOT_FOLDER = "../checkpoint/NIPS2022"
ROOT_FOLDER = "../checkpoint/WWW2023"
# DATASET_LIST = ["alipay", "amazon"]
# DATASET_LIST = ["amazon"]
# DATASET_LIST = ["amazon_beauty", "amazon_electronic"]
# DATASET_LIST = ["movielens", "movielens_100k", "amazon_beauty", "amazon_electronic"]
# DATASET_LIST = ["movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["alipay", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["alipay", "amazon_beauty", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["taobao", "amazon_beauty", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_clothing", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
# DATASET_LIST = ["movielens", "movielens_100k"]
# DATASET_LIST = ["amazon_beauty"]
# MODEL_LIST = ["din", "gru4rec", "sasrec"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]
# TYPE_LIST = ["base", "meta", "meta_hyper_attention"]
# TYPE_LIST = ["base", "meta"]
TYPE_LIST = ["base", "meta", "meta_gru", "meta_ood", "meta_ood_gru"]
# log_filename = "log.txt"
log_filename = "log_ood.txt"
epoch = 10
# result_dict = defaultdict(list)
result_dict = {}
result_dict_ = {}
result_dict2 = {}
result_dict3 = {}
result_dict4 = {}
result_dict5 = {}
result_dict6 = {}
result_dict7 = {}
result_dict8 = {}
result_file = os.path.join(ROOT_FOLDER, "result_ood.txt")
result_file2 = os.path.join(ROOT_FOLDER, "result2_ood.txt")
with open(result_file, "w+") as writer, open(result_file2, "w+") as writer2:
    for dataset in DATASET_LIST:
        print("=" * 50, file=writer)
        print(dataset, file=writer)
        print("-" * 50, file=writer)
        print("=" * 50, file=writer2)
        print(dataset, file=writer2)
        print("-" * 50, file=writer2)
        auc_per10epoch_list = []
        auc_user_per10epoch_list = []
        logloss_per10epoch_list = []
        ndcg5_per10epoch_list = []
        ndcg10_per10epoch_list = []
        ndcg20_per10epoch_list = []
        hr5_per10epoch_list = []
        hr10_per10epoch_list = []
        hr20_per10epoch_list = []
        for model in MODEL_LIST:
            for type in TYPE_LIST:
                # times_num = 0
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                # if type == "meta" or type == "meta_gru":
                #     log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log.txt")
                if not os.path.exists(log_file):
                    continue
                auc_per10epoch_list = []
                auc_user_per10epoch_list = []
                logloss_per10epoch_list = []
                ndcg5_per10epoch_list = []
                hr5_per10epoch_list = []
                ndcg10_per10epoch_list = []
                hr10_per10epoch_list = []
                ndcg20_per10epoch_list = []
                hr20_per10epoch_list = []

                auc_max_list = []
                auc_user_max_list = []
                logloss_max_list = []
                ndcg5_max_list = []
                ndcg10_max_list = []
                ndcg20_max_list = []
                hr5_max_list = []
                hr10_max_list = []
                hr20_max_list = []

                result_dict = {}
                result_dict2 = {}
                result_dict3 = {}
                result_dict4 = {}
                with open(log_file, "r+") as reader:
                    for index, line in enumerate(reader, 1):
                        # print(line.strip("\n"))
                        auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                        auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                        logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                        ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                        hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                        ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                        hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                        ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                        hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])
                        epoch_num = int(line.strip("\n").split(",")[0].split("=")[-1])
                        # print(auc, logloss, ndcg5, hr5)
                        # if epoch_num > epoch:
                        #     continue
                        auc_per10epoch_list.append(auc)
                        auc_user_per10epoch_list.append(auc_user)
                        if not math.isinf(logloss) and not math.isnan(logloss):
                            logloss_per10epoch_list.append(logloss)
                        # logloss_per10epoch_list.append(logloss)
                        ndcg5_per10epoch_list.append(ndcg5)
                        ndcg10_per10epoch_list.append(ndcg10)
                        ndcg20_per10epoch_list.append(ndcg20)
                        hr5_per10epoch_list.append(hr5)
                        hr10_per10epoch_list.append(hr10)
                        hr20_per10epoch_list.append(hr20)

                        # if epoch_num == 1:
                        #     auc_max_list.append(max(auc_per10epoch_list))
                        #     auc_per10epoch_list = []

                        # if index % 10 == 0:

                        # if index % epoch == 0 or not reader.read(1):
                        if index % epoch == 0:
                            # print(auc_per10epoch_list)
                            # print(logloss_per10epoch_list)
                            # print(ndcg5_per10epoch_list)
                            # print(hr5_per10epoch_list)
                            auc_max_list.append(max(auc_per10epoch_list))
                            auc_user_max_list.append(max(auc_user_per10epoch_list))
                            # logloss_max_list.append(max(logloss_per10epoch_list))
                            logloss_max_list.append(min(logloss_per10epoch_list))
                            ndcg5_max_list.append(max(ndcg5_per10epoch_list))
                            ndcg10_max_list.append(max(ndcg10_per10epoch_list))
                            ndcg20_max_list.append(max(ndcg20_per10epoch_list))
                            hr5_max_list.append(max(hr5_per10epoch_list))
                            hr10_max_list.append(max(hr10_per10epoch_list))
                            hr20_max_list.append(max(hr20_per10epoch_list))
                            # print(auc_max_list)
                            # print(logloss_max_list)
                            # print(ndcg5_max_list)
                            # print(hr5_max_list)
                            auc_per10epoch_list = []
                            auc_user_per10epoch_list = []
                            logloss_per10epoch_list = []
                            ndcg5_per10epoch_list = []
                            hr5_per10epoch_list = []
                            ndcg10_per10epoch_list = []
                            hr10_per10epoch_list = []
                            ndcg20_per10epoch_list = []
                            hr20_per10epoch_list = []

                    if len(auc_per10epoch_list) != 0:
                        print(dataset, model, type, len(auc_per10epoch_list), sep=" ")
                        auc_max_list.append(max(auc_per10epoch_list))
                        auc_user_max_list.append(max(auc_user_per10epoch_list))
                        logloss_max_list.append(min(logloss_per10epoch_list))
                        ndcg5_max_list.append(max(ndcg5_per10epoch_list))
                        ndcg10_max_list.append(max(ndcg10_per10epoch_list))
                        ndcg20_max_list.append(max(ndcg20_per10epoch_list))
                        hr5_max_list.append(max(hr5_per10epoch_list))
                        hr10_max_list.append(max(hr10_per10epoch_list))
                        hr20_max_list.append(max(hr20_per10epoch_list))

                    # result_dict[model].append([np.average(auc_max_list), np.std(auc_max_list)])
                    # print("{} {} {} {}」".format(dataset, model, type, auc_max_list))
                    result_dict["{} {}".format(model, type)] = [str(round(float(np.average(auc_max_list)), 4)),
                                                                str(round(float(np.std(auc_max_list)), 4))]
                    result_dict_["{} {}".format(model, type)] = [str(round(float(np.average(auc_user_max_list)), 4)),
                                                                str(round(float(np.std(auc_user_max_list)), 4))]
                    result_dict2["{} {}".format(model, type)] = [str(round(float(np.average(logloss_max_list)), 4)),
                                                                str(round(float(np.std(logloss_max_list)), 4))]
                    result_dict3["{} {}".format(model, type)] = [str(round(float(np.average(ndcg5_max_list)), 4)),
                                                                str(round(float(np.std(ndcg5_max_list)), 4))]
                    result_dict4["{} {}".format(model, type)] = [str(round(float(np.average(hr5_max_list)), 4)),
                                                                str(round(float(np.std(hr5_max_list)), 4))]
                    result_dict5["{} {}".format(model, type)] = [str(round(float(np.average(ndcg10_max_list)), 4)),
                                                                 str(round(float(np.std(ndcg10_max_list)), 4))]
                    result_dict6["{} {}".format(model, type)] = [str(round(float(np.average(hr10_max_list)), 4)),
                                                                 str(round(float(np.std(hr10_max_list)), 4))]
                    result_dict7["{} {}".format(model, type)] = [str(round(float(np.average(ndcg20_max_list)), 4)),
                                                                 str(round(float(np.std(ndcg20_max_list)), 4))]
                    result_dict8["{} {}".format(model, type)] = [str(round(float(np.average(hr20_max_list)), 4)),
                                                                 str(round(float(np.std(hr20_max_list)), 4))]
                    # print(auc_per10epoch_list)
                    # print(logloss_per10epoch_list)
                    # print(ndcg5_per10epoch_list)
                    # print(hr5_per10epoch_list)
                    for key, value in result_dict.items():
                        model, type = key.split(" ")
                        print("{:10s}".format(model), "{:24s}".format(type),
                              "{:12s}".format(" ".join(result_dict["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict_["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict2["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict3["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict4["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict5["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict6["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict7["{} {}".format(model, type)])),
                              "{:12s}".format(" ".join(result_dict8["{} {}".format(model, type)])),
                              # " ".join(result_dict2["{} {}".format(model, type)]),
                              # " ".join(result_dict3["{} {}".format(model, type)]),
                              # " ".join(result_dict4["{} {}".format(model, type)]),
                              sep="\t", file=writer)
                        # if key.split(" ")[-1] != "meta_hyper_attention":
                        #     # print(key)
                        #     print("{:10s}".format(model), "{:24s}".format(type), " ".join(result_dict["{} {}".format(model, type)]), sep=" ", file=writer2)

        print("\n", file=writer)
        print("\n", file=writer2)

    # print("\n\n", file=writer)

