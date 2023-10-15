import torch
import torch.nn as nn


def aggregation(input_moe_tensor, temperature=1, bias=False):
    # if bias:
    #     input_moe_tensor = input_moe_tensor.unsqueeze(2)
    # print("-" * 50)
    # print(input_moe_tensor.size())
    attention_output_activation_func = nn.Softmax(dim=0)
    input_moe_tensor_transpose = torch.transpose(input_moe_tensor, 1, 2)
    # print(input_moe_tensor.size(), input_moe_tensor_transpose.size(), sep="\t")
    # print(torch.matmul(input_moe_tensor_transpose, input_moe_tensor).size())
    # self_attention_matrix = torch.matmul(input_moe_tensor_transpose, input_moe_tensor)
    self_attention_matrix = torch.matmul(input_moe_tensor, input_moe_tensor_transpose)
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(1).size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(1) * input_moe_tensor.size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2).size())
    # print((((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose).size())
    # print((torch.sum(attention_output_activation_func(
    #     (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose, 1).squeeze(1)).size())
    # output_tensor = torch.sum(attention_output_activation_func(
    #     (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose, 1).squeeze(1)
    output_tensor = torch.sum(attention_output_activation_func(
        (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor, 1).squeeze(1)
    return output_tensor


def output_aggregation(input_moe_tensor, temperature=1):
    attention_output_activation_func = nn.Softmax(dim=0)
    input_moe_tensor_transpose = torch.transpose(input_moe_tensor, 1, 2)
    # print(input_moe_tensor.size(), input_moe_tensor_transpose.size(), sep="\t")
    # print(torch.matmul(input_moe_tensor_transpose, input_moe_tensor).size())
    self_attention_matrix = torch.matmul(input_moe_tensor_transpose, input_moe_tensor)
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(1).size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(1) * input_moe_tensor.size())
    # print(((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2).size())
    # print((((torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose).size())
    # print((torch.sum(attention_output_activation_func(
    #     (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose, 1).squeeze(1)).size())
    output_tensor = torch.sum(attention_output_activation_func(
        (torch.sum(self_attention_matrix, 1)) / temperature).unsqueeze(2) * input_moe_tensor_transpose, 1).squeeze(1)
    return output_tensor
