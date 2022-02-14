
import sys
import os
from ray_on_aml.core import Ray_On_AML

#dask

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
import dask
import dask.array as da
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
import mlflow
from azureml.core import Run

dask.config.set(scheduler=ray_dask_get)


def get_data_count():

    abfs = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)

    data = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs)
    return data.count(),ddf.count().compute()

if __name__ == "__main__":
    if ray: #in the headnode
        print("head node detected")
        print("data count result", get_data_count())

    else:
        print("in worker node")
