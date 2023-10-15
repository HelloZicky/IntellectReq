import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_attention(nn.Module):

    def __init__(self, group_num, grad_aggregation_layer):
        super(gradient_attention, self).__init__()
        self.group_num = group_num
        self.attention_output_activation_func = nn.Softmax(dim=0)
        self.grad_aggregation_layer = grad_aggregation_layer

    def self_attention_unit(self, gradient_dict, one_tensor):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        self_attention_gradient_dict = {}
        temperature = 100
        for name, grad_tensor in gradient_dict.items():
            if name in self.grad_aggregation_layer:
                grad_tensor_transpose = torch.transpose(grad_tensor, 0, 1)
                self_attention_matrix = torch.matmul(grad_tensor, grad_tensor_transpose)

                self_attention_gradient_dict[name] = torch.sum(self.attention_output_activation_func(
                    (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(1) * grad_tensor, 0)

        return self_attention_gradient_dict



