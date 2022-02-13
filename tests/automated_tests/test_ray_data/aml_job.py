import ray
import numpy as np
from azureml.core import Run
import os
print("curretn directory ", os.listdir())
import sys
# sys.path.append("../../") # go to parent dir
sys.path.insert(0, './') # location of src 

from src.ray_on_aml.core import Ray_On_AML

import raydp

#dask

if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace
    account_key = ws.get_default_keyvault().get_secret("adls7-account-key")
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()
    print("raydp version ", raydp.__version__)

    if ray: #in the headnode
        additional_spark_configs= {"fs.azure.account.key.adlsdatalakegen6.blob.core.windows.net":"AcDil/MwM9KlDvJu0LBcBIQxogAncv306NMRYABtjphXfWgaDTV3yjZgoSNckUb/3nhG04ND2Nqn553fq36Pqw==",
          "fs.azure.account.key.adlsdatalakegen6.dfs.core.windows.net":"AcDil/MwM9KlDvJu0LBcBIQxogAncv306NMRYABtjphXfWgaDTV3yjZgoSNckUb/3nhG04ND2Nqn553fq36Pqw=="}
        spark = ray_on_aml.getSpark(executor_cores =3,num_executors =3 ,executor_memory='10GB', additional_spark_configs=additional_spark_configs)

        for _ in range(100):
            adls_data = spark.read.format("delta").load("wasbs://mltraining@adlsdatalakegen6.blob.core.windows.net/ISDWeatherDelta")
            adls_data.groupby("stationName").count().head(100)
#         time.sleep(3000)
        



    else:
        print("in worker node")