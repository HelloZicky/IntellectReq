import torch
# import numpy

t1 = torch.Tensor([[1, 15, 16], [2, 16, 17], [13, 3, 10], [13, 4, 10]])
t2 = torch.Tensor([1, 2, 3])
# t2 = torch.Tensor([1, 3, 2])
# t2 = torch.Tensor([2, 1, 3, ])
# t2 = torch.Tensor([3, ])
l2 = [1, 2, 3]
print(t1.size())
print(t2.size())
print(t1[:, 0])
print(t2.any())
print(t2.all())
print(t2.view(-1, 1).any())
print(1 in t2)
# print(t2.item())
# t3 = torch.where(t1[:, 0].view(-1, 1) == t2.any(), True, False)
# t3 = torch.where(t1[:, 0].view(-1, 1) == t2.view(-1, 1), True, False)
# t3 = torch.where(t1[:, 0].view(-1, 1) == t2.view(-1).repeat(t1.size()[0], 1), True, False)
# t3 = torch.where(torch.sum(torch.where(t1[:, 0].view(-1, 1) == t2.view(-1).repeat(t1.size()[0], 1), 1, 0), dim=-1) == 1, True, False)
# t3 = torch.where(torch.sum(torch.where(t1[:, 0].view(-1, 1) == t2.view(-1).repeat(t1.size()[0]), 1, 0), dim=-1) == 1, True, False)
print("-" * 50)
print(t2.view(-1).repeat(t1.size()[0]).size())
t3 = torch.where(torch.sum(torch.where(t1[:, 0].view(-1, 1) == t2.view(-1).repeat(t1.size()[0]), 1, 0), dim=-1) == 1, True, False)
# t3 = torch.where(t1[:, 0] == t2.item(), True, False)
print(t3)
print(set(t1))
print(set(t1[:, 0].numpy()))
print(torch.Tensor(list(set(t1[:, 0].numpy()))))
# t3 = t1[:, 0] in t2
# print(t3)

# length = 645900
# print(length * 1 // 100)
# print(length * 0.1 // 100)
# print(round(length * 0.1, 0) // 100)
# print(int(length * 0.1) // 100)
# print(int(length * 0.1 // 100))