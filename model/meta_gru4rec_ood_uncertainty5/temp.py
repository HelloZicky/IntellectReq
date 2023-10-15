import torch

if __name__ == '__main__':
    output = torch.Tensor([0.8, 0.2, 0.4, 0.5, 0.6])
    print(output)
    request_index = torch.where(output < 0.6, True, False).view(-1)
    output[request_index] = output[request_index] * output[request_index]
    print(output)