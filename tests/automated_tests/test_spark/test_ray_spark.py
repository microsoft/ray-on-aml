
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import raydp
from src.ray_on_aml.core import Ray_On_AML
import ray
import numpy as np
from azureml.core import Run

#dask
if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay(additional_ray_start_head_args="--temp-dir=outputs",additional_ray_start_worker_args="--temp-dir=outputs")

    if ray: #in the headnode
        print("raydp version ", raydp.__version__)
        storage_account_name ="adlsdatalakegen6"
        storage_account_key=ws.get_default_keyvault().get_secret("adlsdatalakegen6")
        additional_spark_configs ={f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net":f"{storage_account_key}"}
        spark = ray_on_aml.getSpark(executor_cores =3,num_executors =2 ,executor_memory='10GB', additional_spark_configs=additional_spark_configs)
        adls_data = spark.read.format("delta").load("abfss://mltraining@adlsdatalakegen6.dfs.core.windows.net/ISDWeatherDelta")
        adls_data.groupby("stationName").count().head(100)