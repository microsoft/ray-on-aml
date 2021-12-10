# Ray on Azure ML

This package simplifies setup of Ray and Ray's components such as DaskOnRay, SparkOnRay, Ray Machine Learning in Azure ML

To install run: pip install --upgrade ray-on-aml

For Interactive use at your compute instance, create a compute cluster in the same vnet where your compute instance is, then run this to get handle to the ray cluster

```python
from ray_on_aml.core import Ray_On_AML
ws = Workspace.from_config()
ray_on_aml =Ray_On_AML(ws=ws, compute_cluster ="worker-cpu-v3")
_, ray = ray_on_aml.getRay()
```

For use in an AML job, include ray_on_aml as a pip dependency and inside your script, do this to get ray
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
Check out examples to learn more 