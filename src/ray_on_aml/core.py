import os
import time
import subprocess
import sys
import platform
import ray
import mlflow
from azureml.core.conda_dependencies import CondaDependencies
from .amlsdkv1_util import get_aml_env_v1, run_ray_exp_run_v1
from .amlsdkv2_util import get_aml_env_v2, run_ray_exp_run_v2
from .generate_entry_script import generate_entry_script
from .util import flush, get_ip
import logging
import zlib
import sys
from ._telemetry._event_logger import _EventLogger

__version__='0.2.4'

module_event_logger = _EventLogger.get_logger(__name__)


class Ray_On_AML():
    """
    Ray_On_AML let you create ray cluster on top of azure ml compute cluster. It handles head node and worker nodes creation
    based on the size you set. It supports 1. Interactive mode where an azure ml compute instance can be used as ray client 
    or a direct head node 2. Job mode where the compute environment of azure ml job is turned into a ray cluster.
    Ray_On_AML relies on Azure ML SDK for interactive mode. You can use either SDK v1 with workspace object or SDK v2 with ml 
    client object. 
    For ray version, in interactive mode, the library requires you to install a ray library at your local machine and it will 
    automatically uses the same ray and python version for the remote cluster creation.
    For job mode, you're responsible to provide ray and other dependencies to the job.
    """

    def __init__(self,compute_cluster=None, ws=None, ml_client=None, exp_name ='ray_on_aml',
                master_ip_env_name="AZ_BATCHAI_MPI_MASTER_NODE", world_rank_env_name="OMPI_COMM_WORLD_RANK",job_timeout=600000, maxnode=2, verbosity=logging.ERROR):
        """
        Init initializes Ray_On_Aml instances to manage life cycle of ray cluster.
        :param compute_cluster: Name of Azure ML cluster created with your Azure ML workspace. In case of interactive use, 
        it must be in the same vnet with your compute instance. vnet is not required for job use.
        :type compute_cluster: str
        :param ws: Azure workspace object in case Azure ML SDK v1 is used. Either ws or ml_client has to be provided for interactive use
        :type ws: ~azureml.core.workspace.Workspace 
        :param ml_client: ML client in case Azure ML SDK v2 is used. Either ws or ml_client has to be provided for interactive use
        :type ml_client: ~azure.ai.ml.MLClient
        :param exp_name: Name of the experiment under which the azure ml job runs for interactive mode
        :type exp_name: str
        :param master_ip_env_name: name of environment variable that contains master node's ip, defaults to AZ_BATCHAI_MPI_MASTER_NODE.
        :type master_ip_env_name: str
        :param world_rank_env_name: name of environment variable that contains world rank of nodes defaults to OMPI_COMM_WORLD_RANK.
        :type world_rank_env_name: str
        :param job_timeout: max duration of the cluster life (azure ml job duration), defaults to 600k seconds.
        :type job_timeout: int
        :param verbosity: defaults to logging.ERROR
        :type verbosity: int

        ---------
        How to run 
        
        ray = ray_on_aml.getRay(num_node=2, pip_packages=["fastparquet==2022.12.0", "azureml-mlflow==1.48.0", "pyarrow==6.0.1", "dask==2022.2.0", "adlfs==2022.11.2", "fsspec==2022.11.0"])
        
        client = ray.init(f"ray://{ray_on_aml.headnode_private_ip}:10001")

        """
        self.ml_client = ml_client #for sdk v2
        self.ws=ws #for sdk v1
        self.compute_cluster=compute_cluster
        self.exp_name= exp_name
        self.master_ip_env_name=master_ip_env_name
        self.world_rank_env_name= world_rank_env_name
        self.num_node=maxnode # deprecated, moved to num_node in getRay() method. Will remove this arg in future 
        self.verbosity = verbosity
        self.job_timeout = job_timeout


        if self.checkNodeType()=="interactive" and ((self.ws is None and self.ml_client is None) or self.compute_cluster is None):
            #Interactive scenario, workspace or ml client is required
            raise Exception("For interactive use, please pass ML client for azureml sdk v2 or workspace for azureml sdk v1 and compute cluster name to the init")

        # SDK V2
        if self.ml_client != None and self.ws == None:
            # check compute_cluster exist 
            try:
                ml_client.compute.get(self.compute_cluster)
            except:
                raise Exception(f"You don't have a cluster named {self.compute_cluster}.") 
        # SDK v1
        elif self.ws != None and self.ml_client == None:
            # check compute_cluster exist            
            try:
                if self.compute_cluster in ws.compute_targets:
                    pass
            except:
                raise Exception(f"You don't have a cluster named {self.compute_cluster}.")
        else:
            raise Exception("You can't use both ml_clinet and ws") 

        if maxnode != None:
            print("Notice:\n\rmaxnode will be deprecated in the future. It will be moved to num_node in getRay() method.")



    def startRayMaster(self, ray_start_head_args):
        try:
            ray.shutdown()
            time.sleep(3)
            ray.shutdown()
        except:
            pass
        
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
                logging.warning("ray start still fails, continue anyway")
        ip = get_ip()
        return ip


    def checkNodeType(self):
        """
        This function checks the environment variable name 'world_rank_env_name'. 
        If it doesn't exist, it means 'interactive' mode. 
        If it does exist, it mean non-interactive mode.
        """

        rank = os.environ.get(self.world_rank_env_name)

        if rank is None:
            return "interactive" # This is interactive scenario
        elif rank == '0':
            return "head"
        else:
            return "worker"


    #check if the current node is headnode
    def startRay(self, additional_ray_start_worker_args, master_ip=None):
        ip = get_ip()
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

        flush(worker_proc, worker_log)

    def getRay(self,ci_is_head=False,environment=None, num_node =2, conda_packages=[],
        pip_packages= [], base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
         shm_size="12g",ray_start_head_args="",ray_start_worker_args="", inputs=None, outputs=None):

        """
        This method execute ray cluster creation on top of azure ml compute cluster.
        :param ci_is_head: Whether to use the current Compute Instance environment as head node, defaults False, in which case this returns the client IP for you to 
        initialize ray client.
        :type conda_packages: [str]
        :param environment: [Appliable for interactive use only] Azure ML environment object, work for either SDK v1 or v2. This is only used when you need to customize 
        ray run time environment for advanced use cases. If you want to run GPU cluster, specify a pre-created GPU environment here
        :type environment: ~azure.ai.ml.entities.Environment or ~azureml.core.environment.Environment or str
        :param conda_packages: [Appliable for interactive use only] list of conda packages to add to the cluster
        :type conda_packages: [str]
        :param pip_packages: [Appliable for interactive use only] list of pip packages to add to the cluster. This is where you can customize ray packages such ["ray[air]==2.1.0","ray[rllib]"]
        if you do not provide a ray package in pip, the package will take the version of the ray in your compute instance and add "ray[default]==YOUR_VERSION" 
        to the list.
        :type pip_packages: [str]
        :param base_image: [Appliable for interactive use only] base image of the azure ml environment on which ray runs, defaults to mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
        :type base_image: str
        :param shm_size: shm_size for the cluster environment, defaults to 8g, can be adjusted to increase the shm_size required by Ray.
        :type shm_size: str
        :param ray_start_head_args: Additional argument values to start ray head. Defaults to empty ""
        :type ray_start_head_args: str
        :param ray_start_worker_args: Additional argument values to start ray workders.Defaults to empty ""
        :type ray_start_worker_args: str
        Return
        ----------
            Returns Ray object
        """
        self.environment = environment
        self.num_node=num_node
        self.conda_packages=conda_packages
        self.pip_packages=pip_packages

        if self.checkNodeType()=="interactive":
            return self.getRayInteractive(ci_is_head, environment, conda_packages, pip_packages, base_image, shm_size,
                                            ray_start_head_args,ray_start_worker_args, inputs, outputs, self.verbosity)
        elif self.checkNodeType() =='head':
            logging.info(f"head node detected, starting ray with head start args {ray_start_head_args}")
            _EventLogger.track_event(module_event_logger, "getRay",{"run_mode":"job"})
            self.startRayMaster(ray_start_head_args)
            time.sleep(10) # wait for the worker nodes to start first
            # ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True )
            return ray
        else:
            logging.info(f"workder node detected , starting ray with worker start args {ray_start_worker_args}")
            self.startRay(ray_start_worker_args)


    def getRayEnvironment(self, environment, conda_packages, pip_packages, base_image):
        """Manager Azure Machine Learning Environement 
        If 'rayEnv__version__' exists in Azure Machine Learning Environment, use the existing one.
        If not, create new one and register it in AML workspace.
        Return
        ----------
        Returns an object of Azure Machine Learning Environment.
        """
        python_version = ["python="+platform.python_version()]
        conda_packages = python_version+conda_packages 

        if "ray" not in str(self.pip_packages):
            try:
                rayVer = f"ray[default]=={ray.__version__}"
                print(rayVer, f"is found in your environment.\n\rRay {ray.__version__} version will be install in your compute cluster")
                self.pip_packages = [rayVer]+ self.pip_packages
            except:
                raise Exception("Ray is not installed")

        if "azureml-mlflow" not in str(self.pip_packages):
            mlflowVer = f"azureml-mlflow"
            print(mlflowVer, "will be added")
            self.pip_packages = [mlflowVer]+ self.pip_packages
        envPkgs = conda_packages + self.pip_packages 
        shEnvPkgs= zlib.adler32(bytes(str(envPkgs),"utf8"))

        if environment:
            envName = environment
        else:
            envName = f"ray-on-aml-{shEnvPkgs}"

        # Find the correct type for sdk v1 and v2
        if type(envName) !=str: #in case user pass a custom environment object, just return it
            return envName

        #this conda dep is common to both v1 and v2. In the futre, if v1 is completely deprecated then this needs to be updated    
        conda_dep = CondaDependencies()
        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        for pip_package in self.pip_packages:
            conda_dep.add_pip_package(pip_package)
        
        # in case SDK v2 is used
        if self.ml_client is not None:
            return get_aml_env_v2(self.ml_client, envName, conda_dep, base_image)
        else: # in case SDK v1 is used   
            return get_aml_env_v1(self.ws, envName, base_image, conda_dep)


    def getRayInteractive(self, ci_is_head, environment,conda_packages, pip_packages, base_image,
                            shm_size, ray_start_head_args, ray_start_worker_args, inputs, outputs, verbosity):
        """Create Compute Cluster, an entry script and Environment
        Create Compute Cluster if given name of Compute Cluster doesn't exist in Azure Machine Learning Workspace
        Get Azure Environement for Ray runtime
        Generate entry script to run Ray in Compute Cluster
        If the script run on Compute Cluster successfully, Ray object will be returned.
        Return
        ----------
            Returns Ray object
        """
        # Verify that cluster does not exist already
        #Used to set base ray packages for interactive scenario

        ##Create the source file
        rayEnv = self.getRayEnvironment(environment, conda_packages, pip_packages, base_image)
        
        os.makedirs(".tmp", exist_ok=True)
        source_file = open(".tmp/source_file.py", "w")
        source_file.write(generate_entry_script(self.job_timeout, self.master_ip_env_name, self.world_rank_env_name, ray_start_head_args, ray_start_worker_args))
        source_file.close()

        if self.ml_client is not None: 
            return self.launch_rayjob_v2(ci_is_head, ray_start_head_args, shm_size,rayEnv, inputs, outputs, verbosity)
        else:
            if inputs != None or outputs != None:
                raise Exception ("Ray-on-aml currently doesn't support inputs/outputs using AML SDK v1, try to use v2 for inputs/outputs")
            return self.launch_rayjob_v1(ci_is_head, ray_start_head_args, shm_size, rayEnv, verbosity)


    def launch_rayjob_v1(self,ci_is_head,ray_start_head_args,shm_size,rayEnv,verbosity):
        if ci_is_head:
            master_ip = self.startRayMaster(ray_start_head_args)
        else:
            master_ip= "None"

        return run_ray_exp_run_v1(ws=self.ws, compute_cluster=self.compute_cluster, num_node=self.num_node, 
        exp_name=self.exp_name, ci_is_head=ci_is_head, ray_start_head_args=ray_start_head_args, shm_size=shm_size, 
        rayEnv=rayEnv, master_ip=master_ip, verbosity=verbosity)


    def launch_rayjob_v2(self, ci_is_head, ray_start_head_args, shm_size, rayEnv, inputs, outputs, verbosity):
        if ci_is_head:
            master_ip = self.startRayMaster(ray_start_head_args)
        else:
            master_ip= "None"

        ray, self.headnode_private_ip, self.cluster_job, self.mount_points = run_ray_exp_run_v2(ml_client=self.ml_client, ci_is_head=ci_is_head, compute_cluster=self.compute_cluster, 
        num_node=self.num_node, exp_name=self.exp_name, ray_start_head_args=ray_start_head_args, shm_size=shm_size, 
        rayEnv=rayEnv, inputs=inputs, outputs=outputs, master_ip=master_ip, verbosity=verbosity) 
        return ray

    
    def shutdown(self, end_all_runs=False):
        """Stop Ray and Compute Cluster
        Parameters
        ----------
        end_all_runs : bool, default=True
            Stop all your compute cluster by default.
            If you want to stop your own Compute Cluster, you can your following
                ray_on_aml.shutdown(end_all_runs=False)
        """
        def end_all_run():            
            from azureml.core import  Experiment

            exp= Experiment(self.ws, self.exp_name)
            runs = exp.get_runs()
            for run in runs:
                if (run.status =='Running') or (run.status =='Preparing'):
                    print("Canceling active run ", run.id, "in", self.exp_name)
                    run.cancel()

        if end_all_runs:
            if self.ws is not None:
                print("Cancel active AML runs if any")
                end_all_run()
            #todo for SDK v2

        if self.ml_client is not None: 
            self.ml_client.jobs.begin_cancel(self.cluster_job.name)
        else:
            self.run.cancel()


    
