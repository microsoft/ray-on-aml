# Ray on Azure ML

This package simplifies setup of Ray and Ray's components such as DaskOnRay, SparkOnRay, Ray Machine Learning in Azure ML for your data science projects.

## Architecture

![RayOnAML_Interactive_Arch](./images/RayOnAML_Interactive_Arch.png)

## Prerequistes

Before you run sample, please check followings.

### 1. Configure Azure Environment

For Interactive use at your compute instance, create a compute cluster in the same vnet where your compute instance is, then run this to get handle to the ray cluster

Check list
> [ ] Azure Machine Learning Workspace
> 
> [ ] Virtual network/Subnet
>
> [ ] Create Compute Instance in the Virtual Network
> 
> [ ] Create Compute Cluster in the same Virtual Network

### 2. Select kernel 

Use ```azureml_py38``` from ```(Jupyter) Notebook``` in Azure Machine Learning Studio to run following examples. 
> Note: VSCode is not supported yet.


### 3. Install library

```bash
pip install --upgrade ray-on-aml
```

> Installing this library will also install ray[default]==1.9.1, pyarrow>= 5.0.0, dask[complete]==2021.12.0, adlfs==2021.10.0 and fsspec==2021.10.1

### 3. Select kernel 

Use ```azureml_py38``` from ```(Jupyter) Notebook``` in Azure Machine Learning Studio to run following examples. 
> Note: Due to Conda env issue,VSCode is not supported yet when using the compute instance as head node when using with ci_is_head = True in getRay() method 

### 4. Run ray-on-aml
Run in interactive mode in a Compute Instance notebook

```python
from ray_on_aml.core import Ray_On_AML
ws = Workspace.from_config()
ray_on_aml =Ray_On_AML(ws=ws, compute_cluster ="Name_of_Compute_Cluster")
ray = ray_on_aml.getRay() # may take around 7 or more mintues

```
Note that by default, one of the nodes in the remote AML compute cluster is used as head node and the remaining are worker nodes. 
But if you want to use your current compute instance as head node and all nodes in the remote compute cluster as workers 
Then simply specify ci_is_head=True).
To install additional library, use additional_pip_packages and additional_conda_packages parameters.

```python
ray_on_aml =Ray_On_AML(ws=ws, compute_cluster ="d15-v2", additional_pip_packages=['torch==1.10.0', 'torchvision', 'sklearn'], maxnode=4)
ray = ray_on_aml.getRay(ci_is_head=True)
```
Advanced usage:There are two arguments to Ray_On_AML() object initilization with to specify base configuration for the library with following default values.
Although it's possible, you should not change the default values of base_conda_dep  and base_pip_dep as it may break the package. Only do so when you need to customize the
cluster default configuration such as ray version.

```python
Ray_On_AML(ws=ws, compute_cluster ="Name_of_Compute_Cluster",base_conda_dep =['adlfs==2021.10.0','pip'],base_pip_dep = ['ray[tune]==1.9.1', 'xgboost_ray==0.1.5', 'dask==2021.12.0','pyarrow >= 5.0.0','fsspec==2021.10.1'])
```

For use in an Azure ML job, include ray_on_aml as a pip dependency and inside your script, do this to get ray
```python

from ray_on_aml.core import Ray_On_AML
ray_on_aml =Ray_On_AML()
ray = ray_on_aml.getRay()

if ray: #in the headnode
    pass
    #logic to use Ray for distributed ML training, tunning or distributed data transformation with Dask

else:
    print("in worker node")
```
### 5. Shutdown ray cluster

To shutdown cluster you must run following.
```ptyhon
ray_on_aml.shutdown()
```

Check out [quick start examples](./examples/quick_use_cases.ipynb) to learn more 
