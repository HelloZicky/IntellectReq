import random
import time
import torch

time_total = 0
# print(time_total)
# n = 1000
n = 100
for i in range(n):
    if i < 10:
        continue
    print(i)
    tensor = torch.Tensor([0.7])
    start_time = time.time()
    threshold = random.random()
    # print(threshold)
    request = tensor > threshold
    end_time = time.time()
    run_time = end_time - start_time
    # print(run_time)
    time_total += run_time

time_total = time_total / n
print(time_total)
