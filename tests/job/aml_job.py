
import mlflow
import ray
import numpy as np
from azureml.core import Run
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray_on_aml import Ray_On_AML
#dask

from ray.util.dask import ray_dask_get
import dask.array as da
import dask.dataframe as dd
import numpy as np
import dask
from adlfs import AzureBlobFileSystem

dask.config.set(scheduler=ray_dask_get)

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
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

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


def get_data_count(account_key):
    account_name="adlsgen7"
    abfs = AzureBlobFileSystem(account_name="adlsgen7",account_key=account_key,  container_name="mltraining")
    abfs2 = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options={'account_name': account_name, 'account_key': account_key}
    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)

    data = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs2)
    data2 = ray.data.read_parquet("az://mltraining/ISDWeatherDelta/year2008", filesystem=abfs)
    return data.count(), data2.count(),ddf.count().compute()
    
if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace
    account_key = ws.get_default_keyvault().get_secret("adls7-account-key")
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray: #in the headnode
        print("head node detected")

        datasets.MNIST("~/data", train=True, download=True)

        analysis = tune.run(train_mnist, config=search_space)
        print(ray.cluster_resources())
        print("data count result", get_data_count(account_key))

    else:
        print("in worker node")