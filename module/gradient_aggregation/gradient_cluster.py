# from sklearn.cluster import DBSCAN
# import numpy as np

# def grad_dbscan(X):
#     y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
#     print("X\n", X)
#     print("\ny_pred\n", y_pred)
#
#
# if __name__ == '__main__':
#     X = np.array([[1, 0],
#                   [2, 0],
#                   [1, 1],
#                   [0, 1],
#                   [0.1, 0.2],
#                   [0.4, 0.3],
#                   [0.5, 0.6],
#                   [0.8, 0.3]])
#     grad_dbscan(X)

from sklearn.cluster import DBSCAN
import numpy as np
import torch


def grad_cluster(group_num, aggregation_layer, gradient_dict):
    # print(X)
    final_gradient_dict = {}
    for name, grad_tensor in gradient_dict.items():
        if name in aggregation_layer:
            grad_norm_matrix = torch.norm(grad_tensor, p=2, dim=1)
            grad_norm_matrix = torch.mm(grad_norm_matrix.unsqueeze(1),
                                        grad_norm_matrix.unsqueeze(0))
            grad_similarity_matrix = torch.mm(grad_tensor, grad_tensor.transpose(0, 1)) / grad_norm_matrix
            for group_index in range(group_num):
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(grad_similarity_matrix[group_index])
                print("=" * 100)
                print(clustering.core_sample_indices_)
                print(clustering.labels_)
                # print(clustering.core_sample_indices_)
                # print(clustering.labels_)
        else:
            # similarity_gradient_dict[name] = torch.zeros(gradient_dict[name][0].size()).to(gradient_dict[name][0].device)
            final_gradient_dict[name] = torch.randn(gradient_dict[name][0].size()).to(gradient_dict[name][0].device) / 100
    # clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
    # clustering = DBSCAN(eps=1, min_samples=2).fit(X)
    # print(clustering.core_sample_indices_)
    #
    # print(clustering.labels_)
    # print(clustering)

    # print(type(torch.from_numpy(clustering.core_sample_indices_).to(X.device)))
    # print(type(torch.from_numpy(clustering.labels_).to(X.device)))


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [9, 9],
                  [25, 80], [26, 80],
                  [29, 82]])
    X = torch.from_numpy(X)
    grad_cluster(X)
