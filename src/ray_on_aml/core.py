import os
import time
import subprocess
import socket
import sys, uuid
import platform
import ray
from textwrap import dedent
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input, MpiDistribution
from azure.ai.ml.entities import Environment
import mlflow
from azureml.core.conda_dependencies import CondaDependencies
import logging
import urllib.request 

__version__='0.2.3'


class Ray_On_AML():
    def __init__(self,compute_cluster=None, ml_client=None, base_conda_dep =['adlfs==2021.10.0','pip==21.3.1'], 
    base_pip_dep = ['ray[default]', 'ray[air]','azureml-mlflow', 'dask==2021.12.0','pyarrow == 6.0.1','fsspec==2021.10.1','fastparquet==0.7.2','tabulate==0.8.9'], 
    vnet_rg = None, vm_size=None, vnet=None, subnet=None,
    exp_name ='ray_on_aml', num_node =2, additional_conda_packages=[],additional_pip_packages=[], job_timeout=600000, master_ip_env_name="AZ_BATCHAI_MPI_MASTER_NODE", world_rank_env_name="OMPI_COMM_WORLD_RANK"):
        """ Class for Ray_On_AML
        Ray_On_AML can help you to minimize your effort for configuring Ray Environment to execute data processing and science tasks on Azure Machine Learning Services.
        Example (AML Compute Instance) : 
        >>> from ray_on_aml.core import Ray_On_AML
        >>> ml_client = Workspace.from_config()
        >>> ray_on_aml =Ray_On_AML(ml_client=ml_client,
        >>>                     exp_name='ray_on_aml',
        >>>                     compute_cluster ="cpu-ray-cluster",
        >>>                     additional_pip_packages=['torch==1.10.0', 'torchvision', 'sklearn'],
        >>>                     maxnode=2)
        Currently this moudle will install following packages for Ray on AML.
            Conda
            *    'adlfs==2021.10.0'
            *    'pip==21.3.1'
            
            Pip
            *    'ray[tune]==1.12.0'
            *    'ray[rllib]==1.12.0'
            *    'dask==2021.12.0'
            *    'pyarrow >= 5.0.0'
            *    'fsspec==2021.10.1'
            *    'fastparquet==0.7.2'
            *    'tabulate==0.8.9'
        This module is compatable with Python 3.7 or higher
        Parameters
        ----------
        ml_client : Workspace
            The Azure Machine Learning Workspace object.
        computer_cluster : string, (optional), default=None
        base_conda_dep : list, default=['adlfs==2021.10.0','pip==21.3.1']
        base_pip_dep : list, default=['ray[tune]==1.12.0','ray[rllib]==1.12.0','ray[serve]==1.12.0', 'xgboost_ray==0.1.8', 'dask==2021.12.0','pyarrow >= 5.0.0','fsspec==2021.10.1','fastparquet==0.7.2','tabulate==0.8.9']
        vnet_rg : string, (optional)
            The default name for Virtual Network will be same as the Resource Group where Azure Machine Leanring workspace is.
        vm_size : string, (optional), default='STANDARD_DS3_V2'        
            The default size for the Compute Cluster is 'STANDARD_DS3_V2'
        vnet : string, (optional)
            The default name of Virtual Network is 'rayvnet'
        subnet : string, (optional)
            The default name of Subnet is 'default'
        exp_name : string, (optional), default='ray_on_aml'
            The name experiment that will be shown in Azure Machine Learning Workspace. 
        num_node : int, (optional), default=5
            You can't change the max number of nodes with this parameter once you created Compute Cluter before.
        additional_conda_packages : list, (optional)
            You can add more package by providing name of packages in a list.         
        additional_pip_packages : list, (optional)
            You can add more package by providing name of packages in a list.
        job_timeout : int, (optional), default=600000
        """
        self.ml_client = ml_client
        self.base_conda_dep=base_conda_dep
        self.base_pip_dep= base_pip_dep
        self.vnet_rg=vnet_rg
        self.compute_cluster=compute_cluster
        self.vm_size=vm_size
        self.vnet=vnet
        self.subnet =subnet
        self.exp_name= exp_name
        self.num_node=num_node
        self.additional_conda_packages=additional_conda_packages
        self.additional_pip_packages=additional_pip_packages
        self.job_timeout = job_timeout
        self.master_ip_env_name=master_ip_env_name
        self.world_rank_env_name= world_rank_env_name


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


    def startRayMaster(self,additional_ray_start_head_args):
        conda_env_name = sys.executable.split('/')[-3]
        logging.info(f"Using {conda_env_name} for the master node")
        cmd =f"ray start --head --port=6379 {additional_ray_start_head_args}"
        try:
            subprocess.check_output(cmd, shell=True)
        except:
            ray_path = f"/anaconda/envs/{conda_env_name}/bin/ray"
            logging.info(f"default ray location is not in PATH, use an alternative path of {ray_path}")
            cmd =f"{ray_path} stop && {ray_path} start --head --port=6379 {additional_ray_start_head_args}"
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

    def getRay(self, logging_level=logging.ERROR, ci_is_head=True, shm_size="8g",base_image =None,gpu_support=False,additional_ray_start_head_args="",additional_ray_start_worker_args=""):
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
        >>>                     maxnode=2)
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
        if self.checkNodeType()=="interactive" and (self.ml_client is None or self.compute_cluster is None):
            #Interactive scenario, workspace object is require
            raise Exception("For interactive use, please pass ML Client and compute cluster name to the init")

        if self.checkNodeType()=="interactive":
            return self.getRayInteractive(logging_level, shm_size,base_image,gpu_support,additional_ray_start_head_args,additional_ray_start_worker_args)
        elif self.checkNodeType() =='head':
            logging.info(f"head node detected, starting ray with additional args {additional_ray_start_head_args}")
            self.startRayMaster(additional_ray_start_head_args)
            time.sleep(10) # wait for the worker nodes to start first
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True )
            return ray
        else:
            logging.info(f"workder node detected , starting ray with additional args {additional_ray_start_worker_args}")
            self.startRay(additional_ray_start_worker_args)



    def getRayEnvironment(self,base_image,gpu_support):
        """Manager Azure Machine Learning Environement 
        If 'rayEnv__version__' exists in Azure Machine Learning Environment, use the existing one.
        If not, create new one and register it in AML workspace.
        Return
        ----------
            Returns an object of Azure Machine Learning Environment.
        """
        python_version = ["python="+platform.python_version()]
        conda_packages = python_version+self.additional_conda_packages +self.base_conda_dep
        pip_packages = self.base_pip_dep +self.additional_pip_packages
        envPkgs = python_version + conda_packages + pip_packages
        shEnvPkgs = abs(hash(str(envPkgs)))
        envName = f"ray-{__version__}-{shEnvPkgs}"

        # if Environment.list(self.ml_client).get(envName) != None:
        #     return Environment.get(self.ml_client, envName)
        # else:
        python_version = ["python="+platform.python_version()]
        conda_packages = python_version+self.additional_conda_packages +self.base_conda_dep
        pip_packages = self.base_pip_dep +self.additional_pip_packages
        
        print(f"Creating new Environment {envName}")
        if not gpu_support:
            if base_image is None:
                base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
        else:
            if base_image is None:
                base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:20221010.v1'
        conda_dep = CondaDependencies()

        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        for pip_package in pip_packages:
            conda_dep.add_pip_package(pip_package)
        conda_dep.save(".tmp/conda.yml")
        rayEnv = Environment(
            image=base_image,
            conda_file=".tmp/conda.yml",
            name="RayEnv",
            description="Environment for ray cluster.",
        )
        # self.ml_client.environments.create_or_update(rayEnv)
        
        return rayEnv


    def getRayInteractive(self, logging_level, shm_size,base_image, gpu_support,additional_ray_start_head_args,additional_ray_start_worker_args):
        """Create Compute Cluster, an entry script and Environment
        Create Compute Cluster if given name of Compute Cluster doesn't exist in Azure Machine Learning Workspace
        Get Azure Environement for Ray runtime
        Generate entry script to run Ray in Compute Cluster
        If the script run on Compute Cluster successfully, Ray object will be returned.
        """
        # Verify that cluster does not exist already

        ##Create the source file
        os.makedirs(".tmp", exist_ok=True)
        rayEnv = self.getRayEnvironment(base_image,gpu_support)


        conda_lib_path = sys.executable.split('/')[-3]+"/lib/python"+sys.version[:3]
        # azureml_py38_PT_TF/lib/python3.8

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
        
            cmd ='ray start --head --port=6379 {4}'
            subprocess.Popen(
            cmd.split(),
            universal_newlines=True
            )
            ip = socket.gethostbyname(socket.gethostname())
            mlflow.log_param("headnode", ip)
            time.sleep({0})
        def checkNodeType():
            rank = os.environ.get("{3}")
            if rank is None:
                return "interactive" # This is interactive scenario
            elif rank == '0':
                return "head"
            else:
                return "worker"
        def startRay(master_ip=None):
            ip = socket.gethostbyname(socket.gethostname())
            print("- env: MASTER_ADDR: ", os.environ.get("{2}"))
            print("- env: RANK: ", os.environ.get("{3}"))
            rank = os.environ.get("{3}")
            master = os.environ.get("{2}")
            print("- my rank is ", rank)
            print("- my ip is ", ip)
            print("- master is ", master)
            if not os.path.exists("logs"):
                os.makedirs("logs")
            print("free disk space on /tmp")
            os.system(f"df -P /tmp")
            if master_ip is None:
                master_ip =master
            cmd = "ray start --address="+master_ip+":6379 {5}"
            print(cmd)
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
        """.format(self.job_timeout,conda_lib_path, self.master_ip_env_name, self.world_rank_env_name,additional_ray_start_head_args,additional_ray_start_worker_args)

        source_file = open(".tmp/source_file.py", "w")
        source_file.write(dedent(source_file_content))
        source_file.close()


        master_ip= "None"




        job = command(
            code=".tmp",
            command="python source_file.py --master_ip ${{inputs.master_ip}}",
            environment=rayEnv,
            inputs={
                "iris_csv": Input(
                    type="uri_file",
                    path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
                ),
                "master_ip": master_ip,
            },
            compute=self.compute_cluster,
            shm_size = shm_size,
            distribution={
                "type": "mpi",
                "process_count_per_instance": 1,},
            instance_count=self.num_node,  

        )

        

        time.sleep(10)


        print("Waiting cluster to start and return head node ip")
        returned_job = self.ml_client.jobs.create_or_update(job)
        headnode_private_ip= None 
        while headnode_private_ip is None:
            headnode_private_ip= mlflow.get_run(run_id=returned_job.id.split("/")[-1]).data.params.get('headnode') 
            print('.', end ="")
            time.sleep(5)
            if mlflow.get_run(run_id=returned_job.id.split("/")[-1])._info.status== 'FAILED':
                print("Cluster startup failed, check detail at run")
                return None, None
        headnode_private_ip = mlflow.get_run(run_id=returned_job.id.split("/")[-1]).data.params.get('headnode')
        print('Headnode has IP:', headnode_private_ip)
        self.headnode_private_ip= headnode_private_ip
        try:
            #disconnect client first to make sure only one is active at a time
            ray.disconnect()
        except:
            pass
        ray.init(f"ray://{headnode_private_ip}:10001",ignore_reinit_error=True, logging_level=logging_level)
        return ray

        
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
        #     exp= Experiment(self.ml_client, self.exp_name)
        #     runs = exp.get_runs()
        #     for run in runs:
        #         if (run.status =='Running') or (run.status =='Preparing'):
        #             print("Canceling active run ", run.id, "in", self.exp_name)
        #             run.cancel()

        # if end_all_runs:
        #     print("Cancel active AML runs if any")
        #     end_all_run()
        # else:
        #     print("Cancel your run")
        #     self.run.cancel()

        try:
            print("Shutting down ray if any")
            self.ray.shutdown()
        except:
            # print("Cannot shutdown ray, ray was not there")
            pass
    
