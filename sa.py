from random import uniform
import torch

batch_size = 4
num_points = 4

half_batch_size = int(batch_size / 2)
normal_sampled = torch.randn(half_batch_size,num_points,3)
uniform_sampled = torch.rand(half_batch_size,num_points,3)
normal_labels = torch.ones(half_batch_size)
uniform_labels = torch.zeros(half_batch_size)

input_data = torch.cat((normal_sampled,uniform_sampled),dim=0)
labels = torch.cat((normal_labels,uniform_labels),dim=0)

data_shuffle = torch.randperm(batch_size)

print("data_shuffle")
print(data_shuffle)

print("input_data")
for i in input_data:
    print(i.view(-1))

print("after")
for i in range(len(data_shuffle)):
    print(input_data[data_shuffle[i]].view(-1))