import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.util.dask import ray_dask_get
from ray_on_aml.core import Ray_On_AML
import dask
import dask.array as da
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
from azureml.core import Run



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


search_space = {
    "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum": tune.uniform(0.01, 0.09)
}

for key in search_space:
    print(key, search_space[key])

model = ConvNet()

optimizer = optim.SGD(model.parameters(), lr=search_space["lr"], momentum=search_space["momentum"])

print(model.parameters())

optim.SGD()



