"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer
from . import self_attention


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i-1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i-1] is not None:
                modules.append(activation_fns[i-1]())

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
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

        return x


class HyperNetwork_FC_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_gru, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )

        self._hyper_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )

        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        # user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)

        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z, _ = self._hyper_gru_cell(z)
            # print("-" * 50)
            # print("z.size() ", z.size())
            if i == 1:
                user_state, _ = self._gru_cell(z)
            else:
                user_state, _ = self._gru_cell(user_state)
            # print("user_state.size() ", user_state.size())  # (512, 10, 32)
            # print("_.size() ", _.size())
            user_state_ = user_state[range(user_state.shape[0]), trigger_seq_length, :]  # (512, 32)
            # print("user_state.size() ", user_state.size())
            z = self._mlp_trans(user_state_)  # (512, 32)
            # print("z.size() ", z.size())
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

        return x


class HyperNetwork_FC_ood(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_ood, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)
        user_state, _ = self._gru_cell(z)
        # print("-" * 50)
        # print(z.size())
        # print(user_state.size())
        # print(user_state.shape[0])
        # print(trigger_seq_length)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            # print("-" * 50)
            # print("i ", i)
            # print("units ", units)
            # print("input_size ", input_size)
            # print("output_size ", output_size)
            # print("x.size() ", x.size())
            # print("z.size() ", z.size())
            # print("self.w1[index].size() ", self.w1[index].size())
            # print("self.b1[index].size() ", self.b1[index].size())
            # print("weight.size() ", weight.size())
            #
            # print("self.w2[index].size() ", self.w2[index].size())
            # print("self.b2[index].size() ", self.b2[index].size())
            # print("bias.size() ", bias.size())
            #
            # print("x.size() ", x.size())
            # print("x.unsqueeze(1).size() ", x.unsqueeze(1).size())
            # print("torch.bmm(x.unsqueeze(1), weight).squeeze(1).size() ", torch.bmm(x.unsqueeze(1), weight).squeeze(1).size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

        return x, user_state


class HyperNetwork_FC_ood_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_ood_gru, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        # user_state, _ = self._gru_cell(z)
        #
        # user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)

        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            if i == 1:
                user_state, _ = self._gru_cell(z)
            else:
                user_state, _ = self._gru_cell(user_state)
            user_state_ = user_state[range(user_state.shape[0]), trigger_seq_length, :]
            z = self._mlp_trans(user_state_)
            
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

        return x, user_state_


class HyperNetwork_FC_hyper_attention(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, moe_num=5, expand=False):
        super(HyperNetwork_FC_hyper_attention, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1, self.w2, self.b1, self.b2 = [], [], [], []
        self.w1_, self.w2_ = [], []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, input_size * output_size).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size, moe_num).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
            # self.w1_.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, output_size).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(output_size, moe_num).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            # self.w2_.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # weight = torch.matmul(z, self_attention.aggregation(self.w1[index])) + self_attention.aggregation(self.b1[index])
            weight = torch.matmul(z, self_attention.aggregation(self.w1[index])) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            # if output_size
            # bias = torch.matmul(z, self_attention.aggregation(self.w2[index])) + self_attention.aggregation(self.b2[index])

            bias = torch.matmul(z, self_attention.aggregation(self.w2[index])) + self.b2[index] \
                if output_size != 1 else torch.matmul(z, self_attention.aggregation(self.w2[index]).unsqueeze(1)) + self.b2[index]

            # print("-" * 50)
            # print(self.w1_[index].size())
            # print(self_attention.aggregation(self.w1[index]).size())
            # print(self.w2_[index].size())
            # print(self_attention.aggregation(self.w2[index]).size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias

        return x


class HyperNetwork_FC_output_attention(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None):
        super(HyperNetwork_FC_output_attention, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        for i in range(1, len(units)):
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        x = self_attention.aggregation(x)
        return x


class HyperNetwork_FC_apg(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, N=64, M=32, K=16):
        super(HyperNetwork_FC_apg, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand

        modules_in = []
        modules_out = []
        modules = []
        # self.net = torch.nn.Sequential(*modules)
        # print("*" * 50)
        # print(units)
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]
            # print("=" * 50)
            linear_in = torch.nn.Linear(input_size, K, bias=True)
            initializer.default_weight_init(linear_in.weight)
            initializer.default_bias_init(linear_in.bias)
            modules_in.append(linear_in)
            # print(linear_in.weight.size())

            linear_out = torch.nn.Linear(K, output_size, bias=True)
            initializer.default_weight_init(linear_out.weight)
            initializer.default_bias_init(linear_out.bias)
            modules_out.append(linear_out)
            # print(linear_out.weight.size())
            # if activation_fns[i - 1] is not None:
            #     modules.append(activation_fns[i - 1]())

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, in_dimension * K * K).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(in_dimension * K * K).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())
            else:
                modules.append(None)

        self.K = K
        # self.modules_in = modules_in
        self.modules_in = torch.nn.Sequential(*modules_in)
        # self.modules_out = modules_out
        self.modules_out = torch.nn.Sequential(*modules_out)
        self.modules = modules

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                # weight = weight.view(input_size, output_size)
                weight = weight.view(self.K, self.K)
            else:
                # weight = weight.view(sample_num, input_size, output_size)
                weight = weight.view(sample_num, self.K, self.K)
            # bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            # print(x.device)
            # print(self.modules_in[index].device)
            # print("-" * 50)
            # print(x.size())
            # print(self.modules_in[index].weight.size())
            x = self.modules_in[index](x)
            # print(x.size())
            # x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1)
            # print(x.size())
            x = self.modules_out[index](x)
            # print(x.size())

            # if self.modules[index] is not None:
            #     x = self.modules[index](x)

        return x