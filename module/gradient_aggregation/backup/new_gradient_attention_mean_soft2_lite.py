import torch
import torch.nn as nn
from torch.nn import functional as F


class gradient_attention_mean(nn.Module):

    def __init__(self, group_num, grad_aggregation_layer):
        super(gradient_attention_mean, self).__init__()
        self.group_num = group_num
        self.attention_output_activation_func = nn.Softmax(dim=0)
        self.grad_aggregation_layer = grad_aggregation_layer

    def similarity_unit(self, similarity_gradient_dict, gradient_dict):
        # exp_item_embedding [batch, 50, embedding_size]
        # candidate_tensor [batch, 50]
        # value_tensor [batch, 50, 1]
        # similarity_gradient_dict = {}
        for name, grad_tensor in gradient_dict.items():
            if name in self.grad_aggregation_layer:
                print("=" * 50)
                mean_grad = torch.mean(gradient_dict[name], 0)
                # mean_grad_similarity_matrix = self.cos_similar(mean_grad.unsqueeze(0), grad_tensor).squeeze(0)
                mean_grad_similarity_matrix = torch.cosine_similarity(mean_grad.unsqueeze(0), grad_tensor).squeeze(0)

                # print(torch.cosine_similarity(mean_grad.unsqueeze(0), grad_tensor).squeeze(0))
                grad_norm_matrix = torch.norm(grad_tensor, p=2, dim=1)
                print("grad_tensor.size() ", grad_tensor.size())
                print("grad_norm_matrix.size() ", grad_norm_matrix.size())
                print("grad_norm_matrix.transpose(0, 1).size() ", grad_norm_matrix.unsqueeze(1).size())
                # print("grad_norm_matrix * grad_norm_matrix.transpose(0, 1).size() ", grad_norm_matrix * grad_norm_matrix.transpose(0, 1).size())
                print("torch.mm(grad_norm_matrix.unsqueeze(1), grad_norm_matrix.unsqueeze(0)).size() ", torch.mm(grad_norm_matrix.unsqueeze(1),
                                                                                                                 grad_norm_matrix.unsqueeze(0)).size())
                print("torch.mm(grad_tensor, grad_tensor.transpose(0, 1)).size() ", torch.mm(grad_tensor, grad_tensor.transpose(0, 1)).size())

                print(torch.mm(grad_tensor, grad_tensor.transpose(0, 1)) / torch.mm(grad_norm_matrix.unsqueeze(1),
                                                                                    grad_norm_matrix.unsqueeze(0)))

                print("mean_grad_similarity_matrix.size()", mean_grad_similarity_matrix.size())

                mean_grad_similarity_matrix_sign = torch.sign(mean_grad_similarity_matrix)

                if torch.sum(mean_grad_similarity_matrix_sign) >= 0:
                    print("-" * 50)
                    # similarity_positive_list = mean_grad_similarity_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)]
                    # similarity_positive_tensor = torch.stack(similarity_positive_list, 0)
                    # mean_grad_similarity_matrix_sign = mean_grad_similarity_matrix_sign.unsqueeze(1)

                    print("mean_grad_similarity_matrix_sign", mean_grad_similarity_matrix_sign)
                    print(torch.where(mean_grad_similarity_matrix_sign > 0))
                    # mean_grad_similarity_matrix_sign.unsqueeze(0)
                    # print("mean_grad_similarity_matrix_sign", mean_grad_similarity_matrix_sign)
                    # print(torch.where(mean_grad_similarity_matrix_sign > 0))

                    common_tensor = grad_tensor[torch.where(mean_grad_similarity_matrix_sign > 0)]
                    # common_tensor = torch.where(mean_grad_similarity_matrix_sign > 0, grad_tensor, torch.zeros(gradient_dict[name][0].size()))
                    print("\nmean_grad_similarity_matrix\n", mean_grad_similarity_matrix)
                    print("\nmean_grad_similarity_matrix_sign\n", mean_grad_similarity_matrix_sign)
                    # print("\ntorch.where(mean_grad_similarity_matrix_sign > 0)\n", torch.where(mean_grad_similarity_matrix_sign > 0))
                    # print("\ntorch.where(mean_grad_similarity_matrix_sign > 0).size()\n", torch.where(mean_grad_similarity_matrix_sign > 0).size())
                    print("\ngrad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)]\n", grad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)])
                    common_tensor_norm_matrix = torch.mm(grad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)].unsqueeze(1),
                                                         grad_norm_matrix[torch.where(mean_grad_similarity_matrix_sign > 0)].unsqueeze(0))
                    # common_tensor_norm_matrix = torch.mm(torch.where(mean_grad_similarity_matrix_sign > 0, grad_norm_matrix, float(-1e5)).unsqueeze(1),
                    #                                      torch.where(mean_grad_similarity_matrix_sign > 0, grad_norm_matrix, float(-1e5)).unsqueeze(0))

                    common_tensor_similarity_matrix = torch.mm(common_tensor, common_tensor.transpose(0, 1)) / common_tensor_norm_matrix
                    # similarity_positive_tensor
                    # similarity_positive_tensor
                    print("\ncommon_tensor.size()\n", common_tensor.size())
                    print("common_tensor_norm_matrix.size()\n", common_tensor_norm_matrix.size())
                    print("common_tensor_similarity_matrix.size()\n", common_tensor_similarity_matrix.size())
                    # gradient_positive_list = grad_tensor[torch.where(mean_grad_similarity_matrix_sign > 0)]
                    # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.stack(similarity_positive_list, 0)).unsqueeze(1) *
                    #                                            torch.stack(gradient_positive_list, 0), 0)
                    # common_tensor_similarity_matrix = torch.where(common_tensor_similarity_matrix > 0, common_tensor_similarity_matrix,
                    #                                               common_tensor_similarity_matrix - 1e5)
                    print("common_tensor_similarity_matrix\n", common_tensor_similarity_matrix)
                    print("\ntorch.sum(common_tensor_similarity_matrix, 1)\n", torch.sum(common_tensor_similarity_matrix, 1))
                    print("\nself.attention_output_activation_func(torch.sum(common_tensor_similarity_matrix, 1)).unsqueeze(1)\n", self.attention_output_activation_func(torch.sum(common_tensor_similarity_matrix, 1)).unsqueeze(1))
                    similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.sum(common_tensor_similarity_matrix, 1)).unsqueeze(1) *
                                                               common_tensor, 0)
                    print("\nsimilarity_gradient_dict[name].size()\n", similarity_gradient_dict[name].size())
                else:
                    # similarity_gradient_dict[name] = torch.randn(gradient_dict[name][0].size()) / 100
                    similarity_gradient_dict[name] = torch.zeros(gradient_dict[name][0].size()) / 100
                # for group_index in range(self.group_num):
                #     pos_num = 0
                #     neg_num = 0
                #     if jump_out:
                #         break
                #     for j in range(self.group_num):
                #         if mean_grad_similarity_matrix[group_index] < 0:
                #             neg_num += 1
                #         elif mean_grad_similarity_matrix[group_index] > 0:
                #             pos_num += 1
                #
                #     if pos_num > neg_num:
                #         gradient_positive_list.append(grad_tensor[group_index])
                #         similarity_positive_list.append(torch.sum(similarity_matrix[group_index]))
                #     elif neg_num >= pos_num:
                #         gradient_negative_list.append(grad_tensor[group_index])
                #         similarity_negative_list.append(torch.sum(similarity_matrix[group_index]))
                # print("=" * 100)
                # print(torch.Tensor(similarity_positive_list).device)
                # print(torch.stack(gradient_positive_list, 0).device)
                # print("torch.stack(similarity_positive_list, 0).size() ", torch.stack(similarity_positive_list, 0).size())
                # print("torch.stack(gradient_positive_list, 0).size() ", torch.stack(gradient_positive_list, 0).size())
                # if len(gradient_positive_list) > len(gradient_negative_list):
                #     # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.Tensor(similarity_positive_list)).unsqueeze(1) *
                #     #                                            torch.stack(gradient_positive_list, 0), 0)
                #     similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.stack(similarity_positive_list, 0)).unsqueeze(1) *
                #                                                torch.stack(gradient_positive_list, 0), 0)
                # else:
                #     # similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.Tensor(similarity_negative_list)).unsqueeze(1) *
                #     #                                            torch.stack(gradient_negative_list, 0), 0)
                #     similarity_gradient_dict[name] = torch.sum(self.attention_output_activation_func(torch.stack(similarity_negative_list, 0)).unsqueeze(1) *
                #                                                torch.stack(gradient_negative_list, 0), 0)
        return similarity_gradient_dict

    # def cos_similar(self, p, q):
    #     sim_matrix = p.matmul(q.transpose(-2, -1))
    #     a = torch.norm(p, p=2, dim=-1)
    #     b = torch.norm(q, p=2, dim=-1)
    #     sim_matrix /= a.unsqueeze(-1)
    #     sim_matrix /= b.unsqueeze(-2)
    #     return sim_matrix



