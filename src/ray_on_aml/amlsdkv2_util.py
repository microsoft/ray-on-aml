import time
import ray
import mlflow
import logging
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

def get_aml_env_v2(ml_client, envName, conda_dep, base_image):
    try:
        return ml_client.environments._get_latest_version(envName)
    except:
        pass

    # If UserError raised
    # Create new Environment
    logging.info(f"Creating new Environment {envName}")
    conda_dep.save(".tmp/conda.yml")

    rayEnv = Environment(
        image=base_image,
        conda_file=".tmp/conda.yml",
        name=envName,
        description="Environment for ray cluster.",
    )

    ml_client.environments.create_or_update(rayEnv)
    return 

def run_ray_exp_run_v2(ml_client, ci_is_head, compute_cluster, num_node, exp_name, ray_start_head_args, shm_size,rayEnv,inputs, outputs, master_ip, verbosity):
        job_input = {"master_ip": master_ip}

        if inputs:
            inputs.update(job_input) 
            job_input = inputs

        job_output = None
        cmd ="python source_file.py"

        for input_key in job_input.keys():
            cmd = cmd + f" --{input_key} ${{{{inputs.{input_key}}}}}"
        if outputs:
            job_output = outputs
            for output_key in job_output.keys():
                cmd = cmd + f" --{output_key} ${{{{outputs.{output_key}}}}}"

        job = command(
            code = ".tmp",
            command = cmd,
            environment = rayEnv,
            inputs = job_input,
            outputs = job_output,
            compute=compute_cluster,
            shm_size = shm_size,
            distribution={
                "type": "mpi",
                "process_count_per_instance": 1,},
            instance_count=num_node,  
            experiment_name = exp_name

        )
        
        cluster_job = ml_client.jobs.create_or_update(job)
        headnode_private_ip= None
        
        # CI as Ray Cluster Head
        if ci_is_head:
            try:
                ray.shutdown()
                time.sleep(3)
                ray.shutdown()
            except:
                pass

            ray.init(address="auto", dashboard_port =5000, ignore_reinit_error=True, logging_level=verbosity)

            print("Waiting for cluster to start")
            waiting = True
            while waiting:
                print('.', end ="")
                status= mlflow.get_run(run_id=cluster_job.id.split("/")[-1])._info.status

                if status == 'FAILED':
                    print("Cluster startup failed, check azure ml run")
                    return None, None
                elif status == 'RUNNING':
                    time.sleep(10)
                    break
                else:
                    time.sleep(5)            
        else:
            print("Waiting cluster to start and return head node's ip")
             
            while headnode_private_ip is None:
                headnode_private_ip= mlflow.get_run(run_id=cluster_job.id.split("/")[-1]).data.params.get('headnode') 
                print('.', end ="")
                time.sleep(5)
                if mlflow.get_run(run_id=cluster_job.id.split("/")[-1])._info.status== 'FAILED':
                    print("Cluster startup failed, check azure ml run")
                    return None
            headnode_private_ip= headnode_private_ip


        params = {}
        max_retry = 20
        count =0
        
        while count<max_retry:
            params = mlflow.get_run(run_id=cluster_job.id.split("/")[-1]).data.params
            count+=1
            time.sleep(1)
            if params !={}:
                break

        if params!={}:
            params.pop("headnode","")
            params.pop("master_ip", "")
            mount_points = params
            if headnode_private_ip:
                print("\n cluster is ready, head node ip ", headnode_private_ip)
            else:
                print("\n Cluster started successfully")
        else:
            print("Cluster startup failed, check azure ml run")
            return None
        
        return ray, headnode_private_ip, cluster_job, mount_points
