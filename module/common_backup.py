"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class HyperNetwork_FC(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=10):
        super(HyperNetwork_FC, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        for i in range(1, len(units)):
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # input_size = units[i - 1]
            output_size = units[i]
            # linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            # initializer.default_weight_init(linear.weight)
            # initializer.default_bias_init(linear.bias)
            # modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        # self.net = torch.nn.Sequential(*modules)
        #
        # self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(), 2))
        # self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(), 2))
        #
        # self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(), 2))
        # self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(), 2))

    def forward(self, x, z, sample_num=32):
        # z.size()          512 10 32
        # z.size().view     512 32*10
        # w1.size()     32*10 (32*2)*128
        # b1.size()       (32*2)*128
        # weight.size()   64 128
        # weight.size()   64 128
        # print("z:", z.size())
        # print("self.w1:", self.w1.size())
        # print("self.w2:", self.w2.size())

        # h_in = torch.matmul(z, self.w2) + self.b2
        # if not self.batch:
        #     h_in = h_in.view(self.in_size, self.z_dim)
        # else:
        #     h_in = h_in.view(sample_num, self.in_size, self.z_dim)
        # # print("h_in.size():", h_in.size())
        # h_final = torch.matmul(h_in, self.w1) + self.b1
        # if not self.batch:
        #     kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
        # else:
        #     kernel = h_final.view(sample_num, self.out_size, self.in_size, self.f_size, self.f_size)
        # param_list = []
        units = self.units
        z = z.view(sample_num, -1)
        for i in range(1, len(units)):
            index = i - 1
            print("-" * 50)
            print("z.size() ", z.size())
            print("self.w1[index].size() ", self.w1[index].size())
            print("self.b1[index].size() ", self.b1[index].size())
            print("torch.matmul(z, self.w1[index]).size() ", torch.matmul(z, self.w1[index]).size())
            print("(torch.matmul(z, self.w1[index]) + self.b1[index]).size() ",
                  (torch.matmul(z, self.w1[index]) + self.b1[index]).size())
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            print("units[i - 1] {}\nunits[i] {}".format(units[i - 1], units[i]))
            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            print("weight.size() ", weight.size())
            print("bias.size() ", bias.size())
            # param_list.append(weight)
            print("x.size() ", x.size())
            # x = F.linear(x, weight, bias)
            print("torch.bmm(x.unsqueeze(1), weight).size() ", torch.bmm(x.unsqueeze(1), weight).size())
            # print("torch.bmm(x, weight).size() ", torch.bmm(x, weight))
            print("(torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias).size() ",
                  (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias).size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

            # h_final = torch.matmul(h_in, self.w2[index]) + self.b2[index]
            # if not self.batch:
            #     kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
            # else:
            #     kernel = h_final.view(sample_num, self.out_size, self.in_size, self.f_size, self.f_size)

        # print("h_final.size():", h_final.size())
        # print("+" * 50)
        # print("kernel.size():", kernel.size())

        # return kernel
        # return param_list
        return x


class HyperNetwork_CNN(nn.Module):
    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
        super(HyperNetwork_CNN, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.batch = batch

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size * self.f_size * self.f_size)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.f_size * self.f_size)).cuda(), 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)).cuda(), 2))

    def forward(self, z, sample_num=32):
        # print("z:", z.size())
        # print("self.w1:", self.w1.size())
        # print("self.w2:", self.w2.size())
        h_in = torch.matmul(z, self.w2) + self.b2
        if not self.batch:
            h_in = h_in.view(self.in_size, self.z_dim)
        else:
            h_in = h_in.view(sample_num, self.in_size, self.z_dim)
        # print("h_in.size():", h_in.size())
        h_final = torch.matmul(h_in, self.w1) + self.b1
        if not self.batch:
            kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
        else:
            kernel = h_final.view(sample_num, self.out_size, self.in_size, self.f_size, self.f_size)

        # print("h_final.size():", h_final.size())
        # print("+" * 50)
        # print("kernel.size():", kernel.size())

        return kernel