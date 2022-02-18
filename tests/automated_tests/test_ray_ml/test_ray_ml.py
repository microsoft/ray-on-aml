
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))

from src.ray_on_aml.core import Ray_On_AML

#dask
from azureml.core import Run
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ray.train.torch
from ray import train
from ray.train import Trainer
from ray import tune
# from ray.tune import Callback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim import Adam
import numpy as np
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256
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


def train_model(model, optimizer, train_loader):
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
        train_model(model, optimizer, train_loader)
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

def train_func(config):
    n = 100
    # create a toy dataset
    # data   : X - dim = (n, 4)
    # target : Y - dim = (n, 1)
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))
    # toy neural network : 1-layer
    # wrap the model in DDP
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=3e-4)
    for epoch in range(config["num_epochs"]):
        y = model.forward(X)
        # compute loss
        loss = criterion(y, Y)
        # back-propagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # To fetch non-DDP state_dict
        # w/o DDP: model.state_dict()
        # w/  DDP: model.module.state_dict()
        # See: https://github.com/ray-project/ray/issues/20915
        state_dict = model.state_dict()
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        train.save_checkpoint(epoch=epoch, model_weights=state_dict)





if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray: #in the headnode
        print("head node detected")
        print("test distributed DL trainining")
        print("resources for ray cluster ", ray.cluster_resources())
        print("torch.cuda.is_available: ", torch.cuda.is_available())


        trainer = Trainer(backend="torch", num_workers=2,use_gpu =True)
        trainer.start()
        trainer.run(train_func, config={"num_epochs": 5})
        trainer.shutdown()

        print(trainer.latest_checkpoint)

        print("test ray tune with pytorch")
        datasets.MNIST("~/data", train=True, download=True)
        #demonstate parallel hyper param tuning
        # Use captureMetrics callback
        # analysis = tune.run(train_mnist, config=search_space, callbacks=[captureMetrics()])
        # run.log_list('acc', accList)
        analysis = tune.run(train_mnist, config=search_space, callbacks=[MLflowLoggerCallback(experiment_name=run.experiment.name, tags={"Framework":"Ray 1.9.1"}, save_artifact=True)])

    else:
        print("in worker node")
