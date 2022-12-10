import os
import time
import subprocess
import socket
import sys, uuid
import platform
import ray
from textwrap import dedent
import mlflow
from azureml.core.conda_dependencies import CondaDependencies
import logging
import zlib
# from warnings import warn

__version__='0.2.3'

#planning
#new feature only supported in v2 SDK
#adding environment object as string
#Adding ability to mount folder
#Adding ability to 
#Set ci_is_head with False default value
# Remove GPU support, instead provide environment as an object. Default is mpi 4.0 as base image
#['adlfs==2021.10.0','pip==21.3.1']
#['ray[default]', 'ray[air]','azureml-mlflow', 'dask==2021.12.0','pyarrow == 6.0.1','fsspec==2021.10.1','fastparquet==0.7.2','tabulate==0.8.9']
class Ray_On_AML():
    def __init__(self,compute_cluster=None, ws=None, ml_client=None, exp_name ='ray_on_aml',base_pip_dep=None,maxnode=2,
                master_ip_env_name="AZ_BATCHAI_MPI_MASTER_NODE", world_rank_env_name="OMPI_COMM_WORLD_RANK"):

        self.ml_client = ml_client #for sdk v2
        self.ws=ws #for sdk v1
        self.compute_cluster=compute_cluster
        self.exp_name= exp_name
        self.master_ip_env_name=master_ip_env_name
        self.world_rank_env_name= world_rank_env_name
        self.num_node=maxnode # deprecated, moved to num_node in getRay() method. Will remove this arg in future 
        try:
            from azure.ai.ml.entities import AmlCompute
            self.azureml_version= "v2"
        except:
            from azureml.core import  Experiment, Environment, ScriptRunConfig, Run
            from azureml.core.runconfig import DockerConfiguration,RunConfiguration
            self.azureml_version= "v1"
        #Used to set base ray packages for interactive scenario
        if not base_pip_dep:
            try:
                self.base_pip_dep = [f"ray[default]=={ray.__version__}"]
            except:
                raise Exception("Ray is not installed")
        else:
            self.ray_packages=base_pip_dep

        if self.checkNodeType()=="interactive" and ((self.ws is None and self.ml_client is None) or self.compute_cluster is None):
            #Interactive scenario, workspace or ml client is required
            raise Exception("For interactive use, please pass ML client for azureml sdk v2 or workspace for azureml sdk v1 and compute cluster name to the init")


    def flush(self,proc, proc_log):
        while True:
            proc_out = proc.stdout.readline()
            if proc_out == "" and proc.poll() is not None:
                proc_log.close()
                break
            elif proc_out:
                sys.stdout.write(proc_out)
                proc_log.write(proc_out)
                proc_log.flush()


    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP


    def startRayMaster(self,ray_start_head_args):
        conda_env_name = sys.executable.split('/')[-3]
        logging.info(f"Using {conda_env_name} for the master node")
        cmd =f"ray start --head --port=6379 {ray_start_head_args}"
        try:
            subprocess.check_output(cmd, shell=True)
        except:
            ray_path = f"/anaconda/envs/{conda_env_name}/bin/ray"
            logging.info(f"default ray location is not in PATH, use an alternative path of {ray_path}")
            cmd =f"{ray_path} stop && {ray_path} start --head --port=6379 {ray_start_head_args}"
            try:
                subprocess.check_output(cmd, shell=True)
            except:
                print("ray start still fails, continue anyway")
        ip = self.get_ip()
        return ip


    def checkNodeType(self):
        rank = os.environ.get(self.world_rank_env_name)
        if rank is None:
            return "interactive" # This is interactive scenario
        elif rank == '0':
            return "head"
        else:
            return "worker"


    #check if the current node is headnode
    def startRay(self,additional_ray_start_worker_args, master_ip=None):
        ip = self.get_ip()
        logging.info(f"- env: MASTER_ADDR: {os.environ.get(self.master_ip_env_name)}")
        rank = os.environ.get(self.world_rank_env_name)
        if master_ip is None:
            master_ip = os.environ.get(self.master_ip_env_name)
        logging.info(f"- my rank is {rank}")
        logging.info(f"- my ip is {ip}")
        logging.info(f"- master is {master_ip}")
        if not os.path.exists("logs"):
            os.makedirs("logs")

        logging.info("free disk space on /tmp")
        os.system(f"df -P /tmp")
        
        cmd = f"ray start --address={master_ip}:6379 {additional_ray_start_worker_args}"

        logging.info(cmd)

        worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")
        return_code=-1
        max_tries =20
        counter =0
        while return_code!=0:
            worker_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            )
            while worker_proc.poll() is None:
                # Process hasn't exited yet, let's wait some
                time.sleep(1)

            # Get return code from process
            return_code = worker_proc.returncode
            
            if return_code!=0:
                logging.warn("Get non zero return code "+str(return_code)+ " retrying after 5s")
                time.sleep(5)
                counter += 1
            else:
                logging.info("Start ray successfully")
            if counter>=max_tries:
                logging.warn("Cannot start ray worker, abort...")
                break

        self.flush(worker_proc, worker_log)

    def getRay(self, logging_level=logging.ERROR,environment=None, num_node =2, conda_packages=[],
        pip_packages= [], base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",job_timeout=600000,
        ci_is_head=True, shm_size="8g",ray_start_head_args="",ray_start_worker_args="", input_data=None):
        """This method automatically creates Azure Machine Learning Compute Cluster with Ray, Dask on Ray, Ray Tune, Ray rrlib, and Ray serve.
        This class takes care of all from infrastructure to runtime preperation, it may take 10 mintues for the first time execution of the module.
        Before you run this method, make sure you have existing Virtual Network and subnet in the same Resource Group where Azure Machine Learning Service is.
        If the Virtual Network is not in the same Resource Group then specify the name of Virtual Network, Subnet name.
        This method can also be used in AML job to turn the remote cluster into Ray cluster.
        
        Example (Interactive use with AML Compute Instance) : 
        >>> from ray_on_aml.core import Ray_On_AML
        >>> ml_client = Workspace.from_config()
        >>> ray_on_aml =Ray_On_AML(ml_client=ml_client,
        >>>                     exp_name='ray_on_aml',
        >>>                     compute_cluster ="cpu-ray-cluster",
        >>>                     additional_pip_packages=['torch==1.10.0', 'torchvision', 'sklearn'],
        >>>                     num_node=2)
        >>> ray = ray_on_aml.getRay()
        Example (use Inside AML Compute Cluster) : 
        >>> from ray_on_aml.core import Ray_On_AML
        >>> ray_on_aml =Ray_On_AML()
        >>> ray = ray_on_aml.getRay()
        >>> if ray: #in the headnode
        >>>     #logic to use Ray for distributed ML training, tunning or distributed data transformation with Dask
        >>> else:
        >>>     print("in worker node")
        Parameters
        ----------
        logging_level : any
            Not implemented yet
        ci_is_head : bool, default=True
            Interactive mode which is using Compute Instant as head is default.
        shm_size : str, default='8g'
            Allow the docker container Ray runs in to make full use of the shared memory available from the host OS. Only applicable for interactive use case
        Return
        ----------
            Returns an object of Ray.        
        """
        self.environment = environment
        self.num_node=num_node
        self.conda_packages=conda_packages
        self.pip_packages=pip_packages
        self.job_timeout = job_timeout

        if self.checkNodeType()=="interactive":
            return self.getRayInteractive(ci_is_head, environment,conda_packages,pip_packages,base_image,shm_size,ray_start_head_args,ray_start_worker_args,logging_level)
        elif self.checkNodeType() =='head':
            logging.info(f"head node detected, starting ray with head start args {ray_start_head_args}")
            self.startRayMaster(ray_start_head_args)
            time.sleep(10) # wait for the worker nodes to start first
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True )
            return ray
        else:
            logging.info(f"workder node detected , starting ray with worker start args {ray_start_worker_args}")
            self.startRay(ray_start_worker_args)



    def getRayEnvironment(self,environment,conda_packages,pip_packages,base_image):
        """Manager Azure Machine Learning Environement 
        If 'rayEnv__version__' exists in Azure Machine Learning Environment, use the existing one.
        If not, create new one and register it in AML workspace.
        Return
        ----------
            Returns an object of Azure Machine Learning Environment.
        """
        python_version = ["python="+platform.python_version()]
        conda_packages = python_version+conda_packages 
        pip_packages = self.base_pip_dep +self.pip_packages
        envPkgs = conda_packages + pip_packages 
        shEnvPkgs= zlib.adler32(bytes(str(envPkgs),"utf8"))
        if environment:
            envName = environment
        else:
            envName = f"ray-on-aml-{shEnvPkgs}"
        if type(envName) !=str: #in case user pass a custom environment object, just return it
            return envName
        #this conda dep is common to both v1 and v2. In the futre, if v1 is completely deprecated then this needs to be updated    
        conda_dep = CondaDependencies()
        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        for pip_package in pip_packages:
            conda_dep.add_pip_package(pip_package)
        if self.ml_client is not None:
            try:
                return self.ml_client.environments._get_latest_version(envName)
            except:
                pass
            print(f"Creating new Environment {envName}")
            conda_dep.save(".tmp/conda.yml")
            print("beging creating env in v2")
            from azure.ai.ml.entities import Environment
            rayEnv = Environment(
                image=base_image,
                conda_file=".tmp/conda.yml",
                name=envName,
                description="Environment for ray cluster.",
            )
            self.ml_client.environments.create_or_update(rayEnv)
        else: # in case skd v1 is used
            from azureml.core.environment import Environment
            if Environment.list(self.ws).get(envName) != None:
                return Environment.get(self.ws, envName)
            rayEnv = Environment(name=envName)
            rayEnv.docker.base_image = base_image
            rayEnv.python.conda_dependencies=conda_dep
            rayEnv.register(self.ws)
            return rayEnv


    def getRayInteractive(self,ci_is_head, environment,conda_packages,pip_packages,base_image, shm_size,ray_start_head_args,ray_start_worker_args,logging_level):
        """Create Compute Cluster, an entry script and Environment
        Create Compute Cluster if given name of Compute Cluster doesn't exist in Azure Machine Learning Workspace
        Get Azure Environement for Ray runtime
        Generate entry script to run Ray in Compute Cluster
        If the script run on Compute Cluster successfully, Ray object will be returned.
        """
        # Verify that cluster does not exist already

        ##Create the source file
        os.makedirs(".tmp", exist_ok=True)
        rayEnv = self.getRayEnvironment(environment,conda_packages,pip_packages,base_image)
        source_file_content = """
        import os
        import time
        import subprocess
        import threading
        import socket
        import sys, uuid
        import platform
        import logging
        from azureml.core import Run
        import ray
        import shutil
        from distutils.dir_util import copy_tree
        import argparse
        import mlflow
        def flush(proc, proc_log):
            while True:
                proc_out = proc.stdout.readline()
                if proc_out == "" and proc.poll() is not None:
                    proc_log.close()
                    break
                elif proc_out:
                    sys.stdout.write(proc_out)
                    proc_log.write(proc_out)
                    proc_log.flush()
        def startRayMaster():
        
            cmd ='ray start --head --port=6379 {3}'
            subprocess.Popen(
            cmd.split(),
            universal_newlines=True
            )
            ip = socket.gethostbyname(socket.gethostname())
            mlflow.log_param("headnode", ip)
            time.sleep({0})
        def checkNodeType():
            rank = os.environ.get("{2}")
            if rank is None:
                return "interactive" # This is interactive scenario
            elif rank == '0':
                return "head"
            else:
                return "worker"
        def startRay(master_ip=None):
            ip = socket.gethostbyname(socket.gethostname())
            logging.info("- env: MASTER_ADDR: ", os.environ.get("{1}"))
            logging.info("- env: RANK: ", os.environ.get("{2}"))
            rank = os.environ.get("{2}")
            master = os.environ.get("{1}")
            logging.info("- my rank is ", rank)
            logging.info("- my ip is ", ip)
            logging.info("- master is ", master)
            if not os.path.exists("logs"):
                os.makedirs("logs")
            logging.info("free disk space on /tmp")
            os.system(f"df -P /tmp")
            if master_ip is None:
                master_ip =master
            cmd = "ray start --address="+master_ip+":6379 {4}"
            logging.info(cmd)
            worker_log = open("logs/worker_"+rank+"_log.txt", "w")
            return_code=-1
            max_tries =20
            counter =0
            while return_code!=0:
                worker_proc = subprocess.Popen(
                cmd.split(),
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                )
                while worker_proc.poll() is None:
                    # Process hasn't exited yet, let's wait some
                    time.sleep(1)
                # Get return code from process
                return_code = worker_proc.returncode
                
                if return_code!=0:
                    logging.warn("Get non zero return code "+str(return_code)+ " retrying after 5s")
                    time.sleep(5)
                    counter += 1
                else:
                    logging.info("Start ray successfully")
                if counter>=max_tries:
                    logging.warn("Cannot start ray worker, abort...")
                    break
            time.sleep({0})
        if __name__ == "__main__":

            parser = argparse.ArgumentParser()
            parser.add_argument("--master_ip")
            args, unparsed = parser.parse_known_args()
            master_ip = args.master_ip
            #check if the user wants CI to be headnode
            if master_ip !="None": 
                startRay(master_ip)
            else:
                if checkNodeType() =="head":
                    startRayMaster()
                else:
                    time.sleep(20)
                    startRay()
        """.format(self.job_timeout, self.master_ip_env_name, self.world_rank_env_name,ray_start_head_args,ray_start_worker_args)

        source_file = open(".tmp/source_file.py", "w")
        source_file.write(dedent(source_file_content))
        source_file.close()
        if self.ml_client is not None: 
            return self.launch_rayjob_v2(ci_is_head,ray_start_head_args,shm_size,rayEnv,logging_level)
        else:
            return self.launch_rayjob_v1(ci_is_head,ray_start_head_args,shm_size,rayEnv,logging_level)



    def launch_rayjob_v1(self,ci_is_head,ray_start_head_args,shm_size,rayEnv,logging_level):
        from azureml.core import  Experiment, Environment, ScriptRunConfig, Run
        from azureml.core.runconfig import DockerConfiguration,RunConfiguration
        from azureml.core.compute import ComputeTarget, AmlCompute

        if ci_is_head:
            master_ip = self.startRayMaster(ray_start_head_args)
        else:
            master_ip= "None"

        docker_config = DockerConfiguration(use_docker=True, shm_size=shm_size)
        aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
        ray_cluster = ComputeTarget(workspace=self.ws, name=self.compute_cluster)

        aml_run_config_ml.target = ray_cluster
        aml_run_config_ml.docker =docker_config
        aml_run_config_ml.node_count = self.num_node
        aml_run_config_ml.environment = rayEnv
        src = ScriptRunConfig(source_directory='.tmp',
                            script='source_file.py',
                            run_config = aml_run_config_ml,
                            arguments = ["--master_ip",master_ip]
                           )

        run = Experiment(self.ws, self.exp_name).submit(src)
        self.run=run

        time.sleep(10)

        if ci_is_head:
            ray.shutdown()
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True, logging_level=logging_level)
            # self.run = run
            # self.ray = ray
            print("Waiting for cluster to start")
            while True:
                active_run = Run.get(self.ws,run.id)
                if active_run.status != 'Running':
                    print('.', end ="")
                    time.sleep(5)
                else:
                    return 
        else:
            print("Waiting cluster to start and return head node ip")
            headnode_private_ip= None 
            while headnode_private_ip is None:
                headnode_private_ip= mlflow.get_run(run_id=run.id).data.params.get('headnode') 
                print('.', end ="")
                time.sleep(5)
                if mlflow.get_run(run_id=run.id)._info.status== 'FAILED':
                    print("Cluster startup failed, check detail at run")
                    return None, None
            self.headnode_private_ip= headnode_private_ip
            print("\n cluster is ready, head node ip ",headnode_private_ip)
            return headnode_private_ip

    def launch_rayjob_v2(self,ci_is_head, ray_start_head_args,shm_size,rayEnv,logging_level):
        from azure.ai.ml import command
        if ci_is_head:
            master_ip = self.startRayMaster(ray_start_head_args)
        else:
            master_ip= "None"

        job = command(
            code=".tmp",
            command="python source_file.py --master_ip ${{inputs.master_ip}}",
            environment=rayEnv,
            inputs={

                "master_ip": master_ip,
            },
            compute=self.compute_cluster,
            shm_size = shm_size,
            distribution={
                "type": "mpi",
                "process_count_per_instance": 1,},
            instance_count=self.num_node,  
            experiment_name = self.exp_name

        )

    

        time.sleep(10)


        print("Waiting cluster to start and return head node ip")
        self.cluster_job = self.ml_client.jobs.create_or_update(job)
        if ci_is_head:
            ray.shutdown()
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True, logging_level=logging_level)
            # self.run = run
            # self.ray = ray
            print("Waiting for cluster to start")
            while True:
                if mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1])._info.status != 'RUNNING':
                    print('.', end ="")
                    time.sleep(5)
                else:
                    return 

        headnode_private_ip= None 
        while headnode_private_ip is None:
            headnode_private_ip= mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1]).data.params.get('headnode') 
            print('.', end ="")
            time.sleep(5)
            if mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1])._info.status== 'FAILED':
                print("Cluster startup failed, check detail at run")
                return None, None
        self.headnode_private_ip= headnode_private_ip
        print("\n cluster is ready, head node ip ",headnode_private_ip)

        return headnode_private_ip

        
    def shutdown(self, end_all_runs=True):
        """Stop Ray and Compute Cluster
        Parameters
        ----------
        end_all_runs : bool, default=True
            Stop all your compute cluster by default.
            If you want to stop your own Compute Cluster, you can your following
                ray_on_aml.shutdown(end_all_runs=False)
        """
        # def end_all_run():
            
        # from azureml.core import  Experiment
        # exp= Experiment(self.ml_client, self.exp_name)
        # runs = exp.get_runs()
        # for run in runs:
        #     if (run.status =='Running') or (run.status =='Preparing'):
        #         print("Canceling active run ", run.id, "in", self.exp_name)
        #         run.cancel()

        # if end_all_runs:
        #     print("Cancel active AML runs if any")
        #     end_all_run()
        # else:
        #     print("Cancel your run")
        #     self.run.cancel()
        if self.ml_client is not None: 
            self.ml_client.jobs.begin_cancel(self.cluster_job.name)
        else:
            self.run.cancel()


    
