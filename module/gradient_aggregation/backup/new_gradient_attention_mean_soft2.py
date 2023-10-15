import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_attention_mean(nn.Module):

    def __init__(self, group_num):
        super(gradient_attention_mean, self).__init__()
        self.group_num = group_num
        similarity_gradient_dict = {}
        self.attention_output_activation_func = nn.Softmax(dim=0)

    def similarity_unit(self, gradient_dict):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        similarity_gradient_dict = {}
        for name, grad_tensor in gradient_dict.items():
            mean_grad = torch.mean(gradient_dict[name], 0)
            # print("mean_grad.size() ", mean_grad.size())
            # print("grad_tensor.size() ", grad_tensor.size())
            # # print("mean_grad.unsqueeze(0).size() ", mean_grad.unsqueeze(0).size())
            # print("self.cos_similar(mean_grad.unsqueeze(0), grad_tensor).size() ", self.cos_similar(mean_grad.unsqueeze(0), grad_tensor).size())
            # print("self.cos_similar(mean_grad.repeat(self.group_num, 1), grad_tensor)\n", self.cos_similar(mean_grad.repeat(self.group_num, 1), grad_tensor))
            # mean_grad_similarity_matrix = self.cos_similar(mean_grad.repeat(self.group_num, 1), grad_tensor)
            mean_grad_similarity_matrix = self.cos_similar(mean_grad.unsqueeze(0), grad_tensor).squeeze(0)

            similarity_matrix = self.cos_similar(grad_tensor, grad_tensor)

            # if name == "attention_layer2.bias":
            #     similarity_matrix = torch.ones(similarity_matrix.size())

            # print("mean_grad_similarity_matrix\n", mean_grad_similarity_matrix)
            # print("similarity_matrix ", similarity_matrix)
            gradient_positive_list = []
            gradient_negative_list = []
            similarity_positive_list = []
            similarity_negative_list = []
            jump_out = False
            for group_index in range(self.group_num):
                pos_num = 0
                neg_num = 0
                if jump_out:
                    break
                for j in range(self.group_num):
                    if mean_grad_similarity_matrix[group_index] < 0:
                        neg_num += 1
                    elif mean_grad_similarity_matrix[group_index] > 0:
                        pos_num += 1

                if pos_num > neg_num:
                    gradient_positive_list.append(grad_tensor[group_index])
                    similarity_positive_list.append(torch.sum(similarity_matrix[group_index]))
                elif neg_num >= pos_num:
                    gradient_negative_list.append(grad_tensor[group_index])
                    similarity_negative_list.append(torch.sum(similarity_matrix[group_index]))

            if len(gradient_positive_list) > len(gradient_negative_list):
                similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.Tensor(similarity_positive_list)).unsqueeze(1) *
                                                           torch.stack(gradient_positive_list, 0), 0)
            else:
                similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.Tensor(similarity_negative_list)).unsqueeze(1) *
                                                           torch.stack(gradient_negative_list, 0), 0)
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



