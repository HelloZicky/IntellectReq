import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import torch

# X = [[-1.1], [0.2], [101.1], [0.3]]
# X = [[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0]]
X = torch.Tensor([[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0]])
# X = [[[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0], [101.1, 1, 0], [0.3, 1, 0]], [[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0]]]
# X = torch.Tensor([[[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0], [101.1, 1, 0], [0.3, 1, 0]], [[-1.1, 1, 0], [0.2, 1, 0], [101.1, 1, 0], [0.3, 1, 0]]])
# clf = LocalOutlierFactor(n_neighbors=3, n_jobs=10)
clf = LocalOutlierFactor(n_neighbors=3)

# for i in range(len(X)):
#     lof_result = clf.fit_predict(X[i])
#     print(lof_result)
lof_result = clf.fit_predict(X)
print(lof_result)