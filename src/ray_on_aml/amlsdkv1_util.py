import time
import ray
import mlflow
from azureml.core import  Experiment, Environment, ScriptRunConfig, Run
from azureml.core.environment import Environment
from azureml.core.runconfig import DockerConfiguration,RunConfiguration
from azureml.core.compute import ComputeTarget

def run_ray_exp_run_v1(ws, compute_cluster, ci_is_head, num_node, exp_name, ray_start_head_args, shm_size, rayEnv, master_ip, verbosity):
    docker_config = DockerConfiguration(use_docker=True, shm_size=shm_size)
    aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
    ray_cluster = ComputeTarget(workspace=ws, name=compute_cluster)

    aml_run_config_ml.target = ray_cluster
    aml_run_config_ml.docker =docker_config
    aml_run_config_ml.node_count = num_node
    aml_run_config_ml.environment = rayEnv
    src = ScriptRunConfig(source_directory='.tmp',
                        script='source_file.py',
                        run_config = aml_run_config_ml,
                        arguments = ["--master_ip",master_ip]
                        )

    run = Experiment(ws, exp_name).submit(src)
    run=run

    time.sleep(10)

    if ci_is_head:
        ray.shutdown()
        ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True, logging_level=verbosity)

        print("Waiting for cluster to start")
        while True:
            active_run = Run.get(ws, run.id)
            if active_run.status != 'Running':
                print('.', end ="")
                time.sleep(5)
            else:
                break 
        print("\n Cluster started successfully")
    else:
        print("Waiting cluster to start and return head node ip")
        headnode_private_ip= None 
        while headnode_private_ip is None:
            headnode_private_ip= mlflow.get_run(run_id=run.id).data.params.get('headnode') 
            print('.', end ="")
            time.sleep(2)
            if mlflow.get_run(run_id=run.id)._info.status== 'FAILED':
                print("Cluster startup failed, check detail at run")
                return None
        # self.headnode_private_ip = headnode_private_ip
        print("\n cluster is ready, head node ip ",headnode_private_ip)

    return ray

def get_aml_env_v1(ws, envName, base_image, conda_dep) -> Environment:
    if Environment.list(ws).get(envName) != None:
        return Environment.get(ws, envName)
        
    rayEnv = Environment(name=envName)
    rayEnv.docker.base_image = base_image
    rayEnv.python.conda_dependencies=conda_dep
    rayEnv.register(ws)
    return rayEnv