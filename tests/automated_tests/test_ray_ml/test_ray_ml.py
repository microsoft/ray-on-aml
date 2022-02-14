
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))

from src.ray_on_aml.core import Ray_On_AML

#dask
from azureml.core import Run

import ray.train.torch
from ray import train
from ray.train import Trainer

import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim import Adam
import numpy as np


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
        print("resources for ray cluster ", ray.cluster_resources())

        trainer = Trainer(backend="torch", num_workers=2)
        trainer.start()
        trainer.run(train_func, config={"num_epochs": 5})
        trainer.shutdown()

        print(trainer.latest_checkpoint)

    else:
        print("in worker node")
