# Ray on Azure ML

This package enables you to use ray and ray's components such as dask on ray, ray[air], ray[data] on top of Azure ML's 
compute instance and compute cluster. With this, you can take advantage of both ray's distributed computing capabilities and Azure machine learning platform. For example you can run ray's distributed ML within AzureML's pipeline and on  managed compute cluster.

With support for both interactive and job uses, you can do interactive development in client/interactive mode then operationalize with job mode.

### [Updates 12/14/2022]


__Support AML SDK v2__

- If you have AML SDK v2 for python in your environment, Ray-On-AML will detect the SDK and leverage AML SDK v2 packages
- This package is still compatable with AML SDK v1.
- If you have both v1 and v2, then v2 will be used as a default.

__Better control of ray versions and ray packages by user__

- Users no longer need to use fixed ray packages that comes with Ray-On-AML. You can specify ray components and versions to use in ``getRay()`` method for interactive mode or include ray version and ray packages in your job [environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?tabs=cli)/dependencies for job mode.

__Ability to mount inputs and outputs to ray cluster (with AML v2) for interactive use__

- No more download or move larger volume of data from Data Lake to compute cluster for processing. Just mounting Data, you can access for read and write data.
- Manage data using Data(Set) in AML, and use the name to mount for in/output
- The path to the mounted folder can be used in ray client for ray to access data.

__Support user define docker environment to greater customize ray environment__

- If you need greater control over the ray's run time environment, you can build the environment using Azure ML's [environment](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.environment?view=azure-python&viewFallbackFrom=azure-ml-py)

## Setup & Quick Start Guide


## Option 1: Run ray workload within an [azure ml job](https://learn.microsoft.com/en-us/cli/azure/ml/job?view=azure-cli-latest) (non-interactive mode)
 1. Setup an [azure ml compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python) 
 2. Include ray-on-aml,azureml-defaults, azureml-mlflow and ray package(s) as job dependencies like below in conda or in your job's [environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?tabs=cli)
 ```
channels:
- anaconda
- conda-forge
dependencies:
- python=3.8.5
- pip:
    - azureml-mlflow
    - azureml-defaults
    - ray-on-aml
    - ray[data]==2.2.0 #add ray packages and versions
    # ..other packages
```
In your job script, you ray cluster handler is available at the head node for you
```
if __name__ == "__main__":
    if ray: #in the headnode
        ray.init(address="auto")
        print(ray.cluster_resources())
        #Your ray logic follows

    else:
        print("in worker node, do nothing")
```
see example at [job](https://github.com/microsoft/ray-on-aml/tree/master/examples/job)


There's no need for vnet setup.

If you like setup an interactive ray cluster to work with from a ray client or directly on the head node, follow the following setup:
## Option 2: Use ray cluster interactively 
You can setup a ray cluster and use it to develop and test interactively either from a head node or with a [ray client](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html)
For this, ray-on-aml relies on an [AML Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) (CI) as the head node or ray client machine and [AML compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python) as a complete remote ray cluster in case the CI is used as ray client only or ray cluster worker(s) in case the CI is used as head node.

## Architecture for Interactive Mode

![RayOnAML_Interactive_Arch](https://github.com/james-tn/ray-on-aml/raw/master/images/RayOnAML_Interactive_Arch.png)

### 1. Setup resources
To setup this mode, you will need a compute instance, compute cluster and they need to be in the same vnet to communicate to each other. Review the following check list
__Checklist for service provisioning__
> [ ] Azure Machine Learning Workspace
> 
> [ ] Virtual network/Subnet
>
> [ ] Network Security Group in/outbound
>
> [ ] Create Compute Instance (CI) in the Virtual Network
> 
> [ ] Create Compute Cluster in the same Virtual Network

### 2. Select kernel 

Use a python 3.7+ conda environment from ```Notebook``` in Azure Machine Learning Studio or ```Jupyter Notebook``` in Azure Machine Learning Compute Instance (CI).

### 3. Install library

Download and install ray-on-aml and ray packages in your notebook conda's environment 

For example, following python command will download and install `ray 2.2.0`, `Azure Machine Learning SDK v2 for python` and other packages
```bash
pip install --upgrade ray==2.2.0 ray[air]==2.2.0 ray[data]==2.2.0 azure-ai-ml ray-on-aml
```

There are two modes to run Ray interactively 

- [Client Mode](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html)
 - Run directly from the head node

### 4.1. Client mode

By default CI won't be part of Ray cluster but it will be used as a terminal to execute job on Ray running on Compute Cluster

```python
from ray_on_aml.core import Ray_On_AML

ray_on_aml =Ray_On_AML(ml_client=ml_client, compute_cluster ="{COMPUTE_CLUSTER_NAME}")

# May take 7 mintues or longer. Check the AML run under ray_on_aml experiment for cluster status.  
ray = ray_on_aml.getRay(num_node=2,pip_packages=["ray[air]==2.2.0","ray[data]==2.2.0","torch==1.13.0","fastparquet==2022.12.0", 
"azureml-mlflow==1.48.0", "pyarrow==6.0.1", "dask==2022.12.0", "adlfs==2022.11.2", "fsspec==2022.11.0"])

client = ray.init(f"ray://{ray_on_aml.headnode_private_ip}:10001")
```

If you ran above sample, make sure you have the same version of ray==2.2.0 in CI.
If you don't specify pip_packages, ray[default] with the same version of ray installed in your CI will be used for the cluster
Behind the scene, an [Azure ML job](https://github.com/microsoft/ray-on-aml/tree/master/examples/job) is launched and create a remote ray cluster that your client connects to.
After this check the resources with ```ray.cluster_resources()``` to see how much resource you have for your ray cluster.
### 4.2. Run at head node

This means CI is setup as header node in the cluster and a remote [azure ml job](https://github.com/microsoft/ray-on-aml/tree/master/examples/job) is launched to provide worker nodes for the cluster . To enable this, set `ci_is_head = True`

```python
from ray_on_aml.core import Ray_On_AML

ray_on_aml =Ray_On_AML(ml_client=ml_client, compute_cluster ="{COMPUTE_CLUSTER_NAME}")

# May take 7 mintues or longer. Check the AML run under ray_on_aml experiment for cluster status.  
# MODE II. CI as Ray cluster Header node
ray = ray_on_aml.getRay(ci_is_head=True, num_node=2)
```
> Note: To install additional library, use ```pip_packages``` and ```onda_packages``` parameters.
> The ray cluster will request 2 nodes from AML if ``num_nodes`` is not specified.

### 5. (AML SDK v2 only) Mount Data(Set) to ray cluster

If you are using AML SDK v2, you can mount Data(Set) to Compute Cluster

```python
from azure.ai.ml import command, Input, Output
from ray_on_aml.core import Ray_On_AML

ray_on_aml =Ray_On_AML(ml_client=ml_client, compute_cluster ="{COMPUTE_CLUSTER_NAME}")

inputs={
    "Input1": Input(
        type="uri_folder",
        path="azureml://datastores/{Data(Set)NAME}/paths/{FolderName}",
    )
}

outputs={
    "Output1": Output(
        type="uri_folder",
        path="azureml://datastores/{Data(Set)NAME}/paths/{FolderName}",
    ),
    "output2": Output(
        type="uri_folder",
        path="azureml://datastores/{Data(Set)NAME}/paths/{FolderName}",
    )
}

ray = ray_on_aml.getRay(inputs = inputs, outputs=outputs, num_node=2,
pip_packages=["ray[air]==2.2.0","ray[data]==2.2.0","torch==1.13.0","fastparquet==2022.12.0", 
"azureml-mlflow==1.48.0", "pyarrow==6.0.1", "dask==2022.2.0", "adlfs==2022.11.2", "fsspec==2022.11.0"])

client = ray.init(f"ray://{ray_on_aml.headnode_private_ip}:10001")
```

### 6. Ray Dashboard

[Only when CI is used as head node ```ci_is_head=True``` ] The easiest way to view Ray dashboard is using the connection from [VSCode for Azure ML](https://code.visualstudio.com/docs/datascience/azure-machine-learning). 
Open VSCode to your Compute Instance then open a terminal, type http://127.0.0.1:8265/ then ctrl+click to open the Ray Dashboard.
![VSCode terminal trick](./images/vs_terminal.jpg)

This trick tells VScode to forward port to your local machine without having to setup ssh port forwarding using VScode's extension on the CI.

![Ray Dashboard](./images/ray_dashboard.jpg)

When running ray in client mode or in job mode with Azure ML cluster, you will need to ssh into the head node and configure port forwarding to view Ray Dashboard

### 7. Shutdown ray cluster

> IMPORTANT: To stop Compute Cluster, you must run shutdown function. And also note that, this function won't stop CI, it only shutdown CC

To shutdown cluster,  run following

```python
ray_on_aml.shutdown()
```

### 8. Specify Ray version and add other Ray and python packages

For Interactive cluster: You can use ```pip_packages``` and ```conda_packages``` arguments in `getRay()` function of the Ray_On_AML object  to configure the ray's run time environment. 
You can also configure your own custom azure ml environment using ``environment`` argument in in `getRay()`.
It can be azureml environmen object or name of the environment.


```python
ray_on_aml =Ray_On_AML(ml_client=ml_client, compute_cluster ="{COMPUTE_CLUSTER_NAME}")

ray = ray_on_aml.getRay(inputs = inputs, outputs=outputs, num_node=2,
pip_packages=["ray[air]==2.2.0","ray[data]==2.2.0","torch==1.13.0","fastparquet==2022.12.0", 
"azureml-mlflow==1.48.0", "pyarrow==6.0.1", "dask==2022.2.0", "adlfs==2022.11.2", "fsspec==2022.11.0"])
```

For Job cluster: simply add ray-on-aml and ray component(s) among other dependencies to your conda file of 
azure ml job or azure ml pipeline.

```python
      - ray-on-aml==0.2.5
      - ray[air]==2.2.0
```

### 9. Quick start examples

Check out [quick start examples](./examples/quick_start_examples.ipynb) to learn more

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Security

Microsoft takes the security of our software products and services seriously, which includes all source code repositories managed through our GitHub organizations, which include [Microsoft](https://github.com/Microsoft), [Azure](https://github.com/Azure), [DotNet](https://github.com/dotnet), [AspNet](https://github.com/aspnet), [Xamarin](https://github.com/xamarin), and [our GitHub organizations](https://opensource.microsoft.com/).

If you believe you have found a security vulnerability in any Microsoft-owned repository that meets [Microsoft's definition of a security vulnerability](https://docs.microsoft.com/en-us/previous-versions/tn-archive/cc751383(v=technet.10)), please report it to us as described below.

## Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them to the Microsoft Security Response Center (MSRC) at [https://msrc.microsoft.com/create-report](https://msrc.microsoft.com/create-report).

If you prefer to submit without logging in, send email to [secure@microsoft.com](mailto:secure@microsoft.com).  If possible, encrypt your message with our PGP key; please download it from the [Microsoft Security Response Center PGP Key page](https://www.microsoft.com/en-us/msrc/pgp-key-msrc).

You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Additional information can be found at [microsoft.com/msrc](https://www.microsoft.com/msrc). 

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

  * Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
  * Full paths of source file(s) related to the manifestation of the issue
  * The location of the affected source code (tag/branch/commit or direct URL)
  * Any special configuration required to reproduce the issue
  * Step-by-step instructions to reproduce the issue
  * Proof-of-concept or exploit code (if possible)
  * Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

If you are reporting for a bug bounty, more complete reports can contribute to a higher bounty award. Please visit our [Microsoft Bug Bounty Program](https://microsoft.com/msrc/bounty) page for more details about our active programs.

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft’s privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

Information on managing Azure telemetry is available at https://azure.microsoft.com/en-us/privacy-data-management/.


## Preferred Languages

We prefer all communications to be in English.

## Policy

Microsoft follows the principle of [Coordinated Vulnerability Disclosure](https://www.microsoft.com/en-us/msrc/cvd).


