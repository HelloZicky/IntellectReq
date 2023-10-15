import os
import numpy
import numpy as np
from collections import defaultdict

ROOT_FOLDER = "../checkpoint/NIPS2022"
# DATASET_LIST = ["alipay", "amazon"]
# DATASET_LIST = ["amazon"]
# DATASET_LIST = ["amazon_beauty", "amazon_electronic"]
# DATASET_LIST = ["movielens", "movielens_100k", "amazon_beauty", "amazon_electronic"]
# DATASET_LIST = ["movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["alipay", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["alipay", "amazon_beauty", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
# DATASET_LIST = ["taobao", "amazon_beauty", "amazon_books", "amazon_sports", "movielens", "movielens_100k", "amazon_electronic"]
DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
# DATASET_LIST = ["movielens", "movielens_100k"]
# DATASET_LIST = ["amazon_beauty"]
# MODEL_LIST = ["din", "gru4rec", "sasrec"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]
# TYPE_LIST = ["base", "meta", "meta_hyper_attention"]
# TYPE_LIST = ["base", "meta"]
TYPE_LIST = ["base", "meta", "meta_ood"]
# log_filename = "log.txt"
log_filename = "log_ood.txt"
epoch = 20
# result_dict = defaultdict(list)
result_dict = {}
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
        for model in MODEL_LIST:
            for type in TYPE_LIST:
                # times_num = 0
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                if type == "meta":
                    log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log.txt")
                if not os.path.exists(log_file):
                    continue
                auc_per10epoch_list = []
                auc_max_list = []
                with open(log_file, "r+") as reader:
                    for index, line in enumerate(reader, 1):
                        # print(line.strip("\n"))
                        auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                        epoch_num = int(line.strip("\n").split(",")[0].split("=")[-1])
                        # if epoch_num > epoch:
                        #     continue
                        auc_per10epoch_list.append(auc)

                        # if epoch_num == 1:
                        #     auc_max_list.append(max(auc_per10epoch_list))
                        #     auc_per10epoch_list = []

                        # if index % 10 == 0:
                        if index % epoch == 0:
                            auc_max_list.append(max(auc_per10epoch_list))
                            auc_per10epoch_list = []

                    if len(auc_per10epoch_list) != 0:
                        print(dataset, model, type, len(auc_per10epoch_list), sep=" ")
                        auc_max_list.append(max(auc_per10epoch_list))

                    # result_dict[model].append([np.average(auc_max_list), np.std(auc_max_list)])
                    # print("{} {} {} {}」".format(dataset, model, type, auc_max_list))
                    result_dict["{} {}".format(model, type)] = [str(round(float(np.average(auc_max_list)), 4)),
                                                                str(round(float(np.std(auc_max_list)), 4))]

        for key, value in result_dict.items():
            model, type = key.split(" ")
            print("{:10s}".format(model), "{:24s}".format(type), " ".join(result_dict["{} {}".format(model, type)]), sep=" ", file=writer)
            if key.split(" ")[-1] != "meta_hyper_attention":
                # print(key)
                print("{:10s}".format(model), "{:24s}".format(type), " ".join(result_dict["{} {}".format(model, type)]), sep=" ", file=writer2)

        print("\n", file=writer)
        print("\n", file=writer2)

    # print("\n\n", file=writer)

