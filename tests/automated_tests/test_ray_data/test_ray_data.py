import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))

from src.ray_on_aml.core import Ray_On_AML


import ray
from ray.util.dask import ray_dask_get,enable_dask_on_ray
import dask
import dask.array as da
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
import mlflow
from azureml.core import Run
print("current directory ", os.getcwd())
# print("home directory ", os.listdir("~"))

print("directory ", os.listdir())

#demonstrate parallel data processing
def get_data_count():

    abfs = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)

    data = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs)
    return data.count(),ddf.count().compute()

if __name__ == "__main__":
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay(additional_ray_start_head_args="--temp-dir=./outputs",additional_ray_start_worker_args="--temp-dir=./outputs")
    enable_dask_on_ray()
    if ray: #in the headnode
        print("head node detected")
        #demonstrate parallel data processing
        print("data count result", get_data_count())

    else:
        print("in worker node")
