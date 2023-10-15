import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class HyperNetwork(nn.Module):

    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.batch = batch

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(), 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(), 2))

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



if __name__ == '__main__':
    from utils import statistics
    model = HyperNetwork()
    statistics.count_param()

