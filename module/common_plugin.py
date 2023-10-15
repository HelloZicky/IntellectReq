"""
Common modules
"""
import torch

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns, add_plugin=False):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)
            if add_plugin and i == len(units) - 1:
                plugin_l1 = torch.nn.Linear(units[i], units[i] // 2, bias=True)
                initializer.default_weight_init(plugin_l1.weight)
                initializer.default_bias_init(plugin_l1.bias)
                modules.append(plugin_l1)

                plugin_l2 = torch.nn.Linear(units[i] // 2, units[i], bias=True)
                initializer.default_weight_init(plugin_l2.weight)
                initializer.default_bias_init(plugin_l2.bias)
                modules.append(plugin_l2)

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

