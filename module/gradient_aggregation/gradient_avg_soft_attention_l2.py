import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_avg_soft_attention_l2(nn.Module):

    def __init__(self, group_num, grad_aggregation_layer):
        super(gradient_avg_soft_attention_l2, self).__init__()
        self.group_num = group_num
        self.attention_output_activation_func = nn.Softmax(dim=0)
        self.grad_aggregation_layer = grad_aggregation_layer

    def similarity_unit(self, gradient_dict):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        similarity_gradient_dict = {}
        grad_attention_weights_list = []
        for name, grad_tensor in gradient_dict.items():
            if name in self.grad_aggregation_layer:
                mean_grad = torch.mean(gradient_dict[name], 0)
                mean_grad_similarity_matrix = torch.cosine_similarity(mean_grad.unsqueeze(0), grad_tensor).squeeze(0)

                grad_norm_matrix = torch.norm(grad_tensor, p=2, dim=1)
                mean_grad_similarity_matrix_sign = torch.sign(mean_grad_similarity_matrix)

                common_tensor = grad_tensor[torch.where(mean_grad_similarity_matrix_sign > 0)]
                common_tensor_norm_matrix = torch.mm(grad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)].unsqueeze(1),
                                                     grad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)].unsqueeze(0))
                common_tensor_similarity_matrix = torch.mm(common_tensor, common_tensor.transpose(0, 1)) / common_tensor_norm_matrix
                grad_attention_weights = self.attention_output_activation_func(
                    torch.sum(common_tensor_similarity_matrix, 1)
                )
                # grad_attention_weights_list.append(grad_attention_weights)
                grad_attention_weights_list.extend(grad_attention_weights)
                similarity_gradient_dict[name] = torch.sum(grad_attention_weights.unsqueeze(1) * common_tensor, 0)
                print("name ", name)
                print("grad_attention_weights.size() ", grad_attention_weights.size())
                print("grad_attention_weights ", grad_attention_weights)
                print("common_tensor_similarity_matrix ", common_tensor_similarity_matrix)
        print("torch.stack(grad_attention_weights_list, 0).size() ", torch.stack(grad_attention_weights_list, 0).size())
        return similarity_gradient_dict, torch.stack(grad_attention_weights_list, 0)


