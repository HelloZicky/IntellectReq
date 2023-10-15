import torch
if __name__ == '__main__':
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [9, 9],
                  [25, 80], [26, 80],
                  [29, 82]])
    X = torch.from_numpy(X)
    grad_cluster(X)