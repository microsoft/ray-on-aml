import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
from src.ray_on_aml.core import Ray_On_AML
import ray
from ray.util.dask import  enable_dask_on_ray
import dask.dataframe as dd
from adlfs import AzureBlobFileSystem
import mlflow
import argparse
import os
import logging
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--datasets", type=str)
    # parse args
    args  = parser.parse_args()

    # return args
    return args

#demonstrate parallel data processing
def get_data_count(path):

    abfs = AzureBlobFileSystem(account_name="azureopendatastorage",  container_name="isdweatherdatacontainer")

    storage_options = {'account_name': 'azureopendatastorage'}
    ddf = dd.read_parquet('az://nyctlc/green/puYear=2019/puMonth=*/*.parquet', storage_options=storage_options)
    data1 = ray.data.read_csv(path+"/iris.csv").repartition(4)
    data2 = ray.data.read_parquet("az://isdweatherdatacontainer/ISDWeather/year=2009", filesystem=abfs)
    return data1.count(),data2.count, ddf.count().compute()

if __name__ == "__main__":
    ray_on_aml =Ray_On_AML(verbosity=logging.INFO)
    ray = ray_on_aml.getRay()
    enable_dask_on_ray()
    args = parse_args()
    print("path to dataset ", args.datasets)
    mlflow.log_param("datasets", args.datasets)
    print( "inside ", os.listdir(args.datasets))

    if ray: #in the headnode
        print("head node detected")
        #demonstrate parallel data processing
        print("data count result", get_data_count(args.datasets))

    else:
        print("in worker node")
