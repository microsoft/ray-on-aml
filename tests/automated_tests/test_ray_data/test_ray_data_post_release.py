from ray_on_aml.core import Ray_On_AML

import ray
from ray.util.dask import ray_dask_get
import dask
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
from azureml.core import Run
from ray.util.dask import ray_dask_get,enable_dask_on_ray

def get_data_count():

    abfs = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)

    data = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs)
    return data.count(),ddf.count().compute()

if __name__ == "__main__":
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()
    enable_dask_on_ray()
    if ray: #in the headnode
        print("head node detected")
        print("data count result", get_data_count())

    else:
        print("in worker node")
