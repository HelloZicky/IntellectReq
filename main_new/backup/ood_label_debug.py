import numpy as np
import torch

pred_a = np.array(
    [
    [0.8, 0.2],  # 1
    [0.4, 0.6],  # 0
    [0.3, 0.7],  # 0
    [0.7, 0.3],  # 1
    [0.3, 0.7],  # 0
    [0.2, 0.8],  # 0
    [0.35, 0.65],  # 0
    [0.65, 0.35]  # 1
]
)
label_a = np.array(
    [
    [0.0],
    [0],
    [0],
    [1],
    [0],
    [0],
    [0],
    [0]
]
)
# print(torch.sigmoid(torch.Tensor(pred_a)))
# print(torch.nn.Softmax(torch.Tensor(pred_a), dim=1))
pred_a = torch.argmax(torch.Tensor(pred_a), dim=1).view(-1, 1)
print(pred_a)
label_a = torch.Tensor(label_a)
ood_label = pred_a == label_a
print(ood_label)
