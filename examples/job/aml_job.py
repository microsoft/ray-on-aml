# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ray
from ray import tune
# from ray.tune import Callback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.util.dask import ray_dask_get
from ray_on_aml.core import Ray_On_AML
import dask
import dask.array as da
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
import mlflow
from azureml.core import Run

dask.config.set(scheduler=ray_dask_get)

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256
run = Run.get_context()
ws = run.experiment.workspace
# set mlflow uri
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(run.experiment.name)
ray_on_aml =Ray_On_AML()
ray = ray_on_aml.getRay()

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


# class captureMetrics(Callback)  :
#     def on_trial_result(self, iteration, trials, trial, result, **info):
#         # accList.append(result['mean_accuracy'])
#         print(f"Got result: {result['mean_accuracy']}")


def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_mnist(config):
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")


search_space = {
    "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum": tune.uniform(0.01, 0.09)
}

#demonstrate parallel data processing
def get_data_count():

    abfs = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)

    data = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs)
    return data.count(),ddf.count().compute()

if __name__ == "__main__":
    if ray: #in the headnode
        print("head node detected")

        datasets.MNIST("~/data", train=True, download=True)
        #demonstate parallel hyper param tuning
        # Use captureMetrics callback
        # analysis = tune.run(train_mnist, config=search_space, callbacks=[captureMetrics()])
        # run.log_list('acc', accList)
        analysis = tune.run(train_mnist, config=search_space, callbacks=[MLflowLoggerCallback(experiment_name=run.experiment.name, tags={"Framework":"Ray 1.9.1"}, save_artifact=True)])

        #demonstrate parallel data processing
        print("data count result", get_data_count())

    else:
        print("in worker node")
