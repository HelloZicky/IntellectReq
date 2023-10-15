import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_attention_mean_normal(nn.Module):

    def __init__(self, group_num):
        super(gradient_attention_mean_normal, self).__init__()
        self.group_num = group_num
        similarity_gradient_dict = {}
        self.attention_output_activation_func = nn.Softmax(dim=0)

    def similarity_unit(self, gradient_dict, gradient_1d_size_dict):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        similarity_gradient_dict = {}
        gradient_positive_dict = {}
        gradient_negative_dict = {}

        for name, grad_tensor in gradient_dict.items():
            # print("\n", "=" * 50, "\n")
            # print(name)
            similarity_matrix = self.cos_similar(grad_tensor, grad_tensor)
            if name == "attention_layer2.bias":
                similarity_matrix = torch.ones(similarity_matrix.size())
            # if name == "attention_layer2.bias":
            #     similarity_matrix = torch.ones(grad_tensor.size())
            # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(
            #     torch.sum(similarity_matrix, 1)).unsqueeze(1) * grad_tensor, 0)
            print("name ", name)
            print("similarity_matrix\n", similarity_matrix)
            jump_out = False
            for group_index in range(self.group_num):
                if jump_out:
                    break
                for j in range(group_index, self.group_num):
                    # if similarity_matrix[group_index, j] < 0:
                    # if similarity_matrix[group_index, j] < 0.5 or torch.isnan(similarity_matrix[group_index, j]):
                    if similarity_matrix[group_index, j] < 0 or torch.isnan(similarity_matrix[group_index, j]):
                        # gradient_dict[name] = torch.randn(gradient_size_dict[name])
                        # print(similarity_gradient_dict[name][group_index].size())
                        # print(torch.randn(gradient_1d_size_dict[name]).size())
                        similarity_gradient_dict[name] = torch.randn(gradient_1d_size_dict[name], requires_grad=False)
                        # similarity_gradient_dict[name] = torch.zeros(gradient_1d_size_dict[name])
                        update = False
                        jump_out = True
                        break

                        # similarity_gradient_dict[name][group_index] = torch.randn(grad_tensor[group_index].size())

                    # elif torch.isnan(similarity_matrix[group_index, j]):
                    #     similarity_gradient_dict[name] = torch.mean(grad_tensor, 0)
                    # elif j == self.group_num - 1 and similarity_matrix[group_index, j] >= 0.5:
                    elif group_index == self.group_num - 1 and j == self.group_num - 1:
                        # gradient_dict[name] = similarity_matrix[group_index] * grad_tensor
                        # print(similarity_matrix[group_index].size())
                        # print(grad_tensor.size())
                        # print((similarity_matrix[group_index].unsqueeze(1) * grad_tensor).size())
                        # print(torch.sum(similarity_matrix[group_index].unsqueeze(1) * grad_tensor, 0).size())
                        # similarity_gradient_dict[name][group_index] = similarity_matrix[group_index].unsqueeze(1) * grad_tensor
                        # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(
                        #     torch.sum(similarity_matrix, 1)).unsqueeze(1) * grad_tensor, 0)
                        similarity_gradient_dict[name] = torch.mean(gradient_dict[name], 0)
                        update = True
        return similarity_gradient_dict

    def cos_similar(self, p, q):
        sim_matrix = p.matmul(q.transpose(-2, -1))
        a = torch.norm(p, p=2, dim=-1)
        b = torch.norm(q, p=2, dim=-1)
        sim_matrix /= a.unsqueeze(-1)
        sim_matrix /= b.unsqueeze(-2)
        return sim_matrix

    def forward(self, gradient_dict, gradient_1d_size_dict):
        similarity_gradient_dict = self.similarity_unit(gradient_dict, gradient_1d_size_dict)
        return similarity_gradient_dict



