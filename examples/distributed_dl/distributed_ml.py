
from ray_on_aml.core import Ray_On_AML
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
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim import Adam
import numpy as np

def train_func(config):
    cuda = torch.device('cuda')
    n = 100
    # create a toy dataset
    # data   : X - dim = (n, 4)
    # target : Y - dim = (n, 1)
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4))).detach().to(cuda)
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1))).detach().to(cuda)
    # toy neural network : 1-layer
    # wrap the model in DDP
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=3e-4)
    for epoch in range(config["num_epochs"]):
        y = model.forward(X)
        # compute loss
        loss = criterion(y, Y)
        print("epoch ", epoch, " loss ", loss)

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
        ray.init(address="auto")
        print("resources for ray cluster ", ray.cluster_resources())


        trainer = Trainer(backend="torch", num_workers=2,use_gpu =True)
        trainer.start()
        trainer.run(train_func, config={"num_epochs": 5})
        trainer.shutdown()

        print(trainer.latest_checkpoint)
    else:
        print("in worker node")
