import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_attention_mean(nn.Module):

    def __init__(self, group_num):
        super(gradient_attention_mean, self).__init__()
        self.group_num = group_num
        similarity_gradient_dict = {}
        self.attention_output_activation_func = nn.Softmax(dim=0)

    def similarity_unit(self, gradient_dict, gradient_1d_size_dict):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        similarity_gradient_dict = {}
        for name, grad_tensor in gradient_dict.items():
            # print("\n", "=" * 50, "\n")
            # print(name)
            similarity_matrix = self.cos_similar(grad_tensor, grad_tensor)
            if name == "attention_layer2.bias":
                similarity_matrix = torch.ones(similarity_matrix.size())


            # print("name ", name)
            # print("similarity_matrix\n", similarity_matrix)
            # if name == "attention_layer2.bias":
            #     similarity_matrix = torch.ones(grad_tensor.size())
            # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(
            #     torch.sum(similarity_matrix, 1)).unsqueeze(1) * grad_tensor, 0)
            gradient_general_list = []
            gradient_positive_list = []
            gradient_negative_list = []
            update = 10
            jump_out = False
            for group_index in range(self.group_num):
                pos_num = 0
                neg_num = 0
                if jump_out:
                    break
                for j in range(self.group_num):
                    # if similarity_matrix[group_index, j] < 0:
                    if similarity_matrix[group_index, j] < 0:
                        neg_num += 1
                    elif similarity_matrix[group_index, j] >= 0:
                        pos_num += 1
                if pos_num == self.group_num or neg_num == self.group_num:
                    gradient_general_list.append(grad_tensor[group_index])
                else:
                    # if max(pos_num, neg_num) > 0.6 * self.group_num:
                    if pos_num > neg_num:
                        gradient_positive_list.append(grad_tensor[group_index])
                    elif neg_num >= pos_num:
                        gradient_negative_list.append(grad_tensor[group_index])
            # print("len(gradient_general_list) ", len(gradient_general_list))
            # print("len(gradient_positive_list) ", len(gradient_positive_list))
            # print("len(gradient_negative_list) ", len(gradient_negative_list))
            if len(gradient_general_list) > 0:
                # print("general")
                # print(gradient_general_list[0].size())
                # print("torch.cat(gradient_general_list.size() ", torch.cat(gradient_general_list).size())
                # print("torch.cat(gradient_general_list, 1).size() ", torch.cat(gradient_general_list, 1).size())
                # print("torch.mean(torch.cat(gradient_general_list), 0) ", torch.mean(torch.cat(gradient_general_list), 0))

                # similarity_gradient_dict[name] = torch.mean(torch.cat(gradient_general_list), 0)
                similarity_gradient_dict[name] = torch.mean(torch.stack(gradient_general_list, 0), 0)
            else:
                # if max(len(gradient_positive_list), len(gradient_negative_list)) > 0.6 * self.group_num:
                if len(gradient_positive_list) > len(gradient_negative_list):
                    # print("positive")
                    # print(gradient_positive_list[0].size())
                    # similarity_gradient_dict = torch.mean(gradient_positive_list, 0)
                    # print("torch.cat(gradient_positive_list.size() ", torch.cat(gradient_positive_list).size())
                    # print("torch.cat(gradient_positive_list, 1).size() ", torch.cat(gradient_positive_list, 1).size())
                    # print("torch.mean(torch.cat(gradient_positive_list), 0) ", torch.mean(torch.cat(gradient_positive_list), 0))
                    similarity_gradient_dict[name] = torch.mean(torch.stack(gradient_positive_list, 0), 0)
                else:
                    # print("positive")
                    # print(gradient_negative_list[0].size())
                    # similarity_gradient_dict = torch.mean(gradient_positive_list, 0)
                    # print("torch.cat(gradient_negative_list.size() ", torch.cat(gradient_negative_list).size())
                    # print("torch.cat(gradient_negative_list, 1).size() ", torch.cat(gradient_negative_list, 1).size())
                    # print("torch.mean(torch.cat(gradient_negative_list), 0) ", torch.mean(torch.cat(gradient_negative_list), 0))
                    similarity_gradient_dict[name] = torch.mean(torch.stack(gradient_negative_list, 0), 0)
            # print("grad_tensor.size() ", grad_tensor.size())
            # print("similarity_gradient_dict[name].size() ", similarity_gradient_dict[name].size())
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



