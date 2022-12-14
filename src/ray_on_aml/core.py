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
from ._telemetry._loggerfactory import _LoggerFactory
# from warnings import warn

__version__='0.2.3'

#planning
#new feature only supported in v2 SDK
#adding environment object as string
#Set ci_is_head with False default value
# Remove GPU support, instead provide environment as an object. Default is mpi 4.0 as base image
#['adlfs==2021.10.0','pip==21.3.1']

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
        ray.shutdown()
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
        """
        self.environment = environment
        self.num_node=num_node
        self.conda_packages=conda_packages
        self.pip_packages=pip_packages

        if self.checkNodeType()=="interactive":
            return self.getRayInteractive(ci_is_head, environment,conda_packages,pip_packages,base_image,shm_size,ray_start_head_args,ray_start_worker_args,inputs, outputs, self.verbosity)
        elif self.checkNodeType() =='head':
            logging.info(f"head node detected, starting ray with head start args {ray_start_head_args}")
            _LoggerFactory.track({"run_mode":"job"})
            self.startRayMaster(ray_start_head_args)
            time.sleep(10) # wait for the worker nodes to start first
            # ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True )
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
        if "ray" not in str(self.pip_packages):
            try:
                self.pip_packages = [f"ray[default]=={ray.__version__}"]+ self.pip_packages
            except:
                raise Exception("Ray is not installed")
        if "azureml-mlflow" not in str(self.pip_packages):
            self.pip_packages = [f"azureml-mlflow"]+ self.pip_packages
        envPkgs = conda_packages + self.pip_packages 
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

        for pip_package in self.pip_packages:
            conda_dep.add_pip_package(pip_package)
        if self.ml_client is not None:
            try:
                return self.ml_client.environments._get_latest_version(envName)
            except:
                pass
            logging.info(f"Creating new Environment {envName}")
            conda_dep.save(".tmp/conda.yml")
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


    def getRayInteractive(self,ci_is_head, environment,conda_packages,pip_packages,base_image, shm_size,ray_start_head_args,ray_start_worker_args,inputs, outputs, verbosity):
        """Create Compute Cluster, an entry script and Environment
        Create Compute Cluster if given name of Compute Cluster doesn't exist in Azure Machine Learning Workspace
        Get Azure Environement for Ray runtime
        Generate entry script to run Ray in Compute Cluster
        If the script run on Compute Cluster successfully, Ray object will be returned.
        """
        # Verify that cluster does not exist already
        #Used to set base ray packages for interactive scenario

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
        import ray
        import shutil
        from distutils.dir_util import copy_tree
        import argparse
        import mlflow
        instrumentation_key = "28f3e437-7871-4f33-a75a-b5b3895438db"
        class _LoggerFactory:
            @staticmethod
            def get_logger(verbosity=logging.INFO):
                logger = logging.getLogger(__name__)
                logger.setLevel(verbosity)
                try:
                    from opencensus.ext.azure.log_exporter import AzureLogHandler

                    if not _LoggerFactory._found_handler(logger, AzureLogHandler):
                        logger.addHandler(
                            AzureLogHandler(
                                connection_string="InstrumentationKey=" + instrumentation_key
                            )
                        )
                except Exception:
                    pass

                return logger

            @staticmethod
            def _found_handler(logger, handler_type):
                for log_handler in logger.handlers:
                    if isinstance(log_handler, handler_type):
                        return True
                return False

            @staticmethod
            def _try_get_run_info():
                try:
                    import re
                    import os
                    import ray
                    location = os.environ.get("AZUREML_SERVICE_ENDPOINT")
                    location = re.compile("//(.*?)\\.").search(location).group(1)
                except Exception:
                    location = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
                return {{
                    "subscription": os.environ.get("AZUREML_ARM_SUBSCRIPTION", ""),
                    "run_id": os.environ.get("AZUREML_RUN_ID", ""),
                    "resource_group": os.environ.get("AZUREML_ARM_RESOURCEGROUP", ""),
                    "workspace_name": os.environ.get("AZUREML_ARM_WORKSPACE_NAME", ""),
                    "experiment_id": os.environ.get("AZUREML_EXPERIMENT_ID", ""),
                    "location": location,
                    "ray_version": ray.__version__,
                }}
            @staticmethod
            def track(info):
                logger = _LoggerFactory.get_logger(verbosity=logging.INFO)
                run_info = _LoggerFactory._try_get_run_info()
                if run_info is not None:
                    info.update(run_info)        
                logger.info(msg=info)
    
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--master_ip")
            # parse args
            args,unknown = parser.parse_known_args()
            for arg in unknown:
                if arg.startswith(("-", "--")):
                    parser.add_argument(arg.split('=')[0])
            args = parser.parse_args()
            return args
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
            if return_code == 0:
                time.sleep({0})
        if __name__ == "__main__":

            args = parse_args()
            master_ip = args.master_ip
            #log mount points (inputs, outputs) to the param so that users can use
            for k,v in args.__dict__.items():
                mlflow.log_param(k, v)
             #check if the user wants CI to be headnode
            if master_ip !="None": 
                _LoggerFactory.track({{'run_mode':'interactive_ci_head'}})
                startRay(master_ip)

            else:
                if checkNodeType() =="head":
                    _LoggerFactory.track({{'run_mode':'interactive_client'}})
                    startRayMaster()
                else:
                    time.sleep(20)
                    startRay()
        """.format(self.job_timeout, self.master_ip_env_name, self.world_rank_env_name,ray_start_head_args,ray_start_worker_args)

        source_file = open(".tmp/source_file.py", "w")
        source_file.write(dedent(source_file_content))
        source_file.close()

        if self.ml_client is not None: 
            return self.launch_rayjob_v2(ci_is_head,ray_start_head_args,shm_size,rayEnv,inputs, outputs, verbosity)
        else:
            return self.launch_rayjob_v1(ci_is_head,ray_start_head_args,shm_size,rayEnv,verbosity)



    def launch_rayjob_v1(self,ci_is_head,ray_start_head_args,shm_size,rayEnv,verbosity):
        from azureml.core import  Experiment, Environment, ScriptRunConfig, Run
        from azureml.core.runconfig import DockerConfiguration,RunConfiguration
        from azureml.core.compute import ComputeTarget

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
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True, logging_level=verbosity)

            print("Waiting for cluster to start")
            while True:
                active_run = Run.get(self.ws,run.id)
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
            self.headnode_private_ip= headnode_private_ip
            print("\n cluster is ready, head node ip ",headnode_private_ip)

        return ray

    def launch_rayjob_v2(self,ci_is_head, ray_start_head_args,shm_size,rayEnv,inputs, outputs, verbosity):
        from azure.ai.ml import command
        if ci_is_head:
            master_ip = self.startRayMaster(ray_start_head_args)
        else:
            master_ip= "None"
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
            code=".tmp",
            command=cmd,
            environment=rayEnv,
            inputs=job_input,
            outputs = job_output,
            compute=self.compute_cluster,
            shm_size = shm_size,
            distribution={
                "type": "mpi",
                "process_count_per_instance": 1,},
            instance_count=self.num_node,  
            experiment_name = self.exp_name

        )
        
        self.cluster_job = self.ml_client.jobs.create_or_update(job)
        headnode_private_ip= None
        if ci_is_head:
            ray.shutdown()
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True, logging_level=verbosity)
            print("Waiting for cluster to start")
            waiting = True
            while waiting:
                print('.', end ="")
                status= mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1])._info.status 
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
                headnode_private_ip= mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1]).data.params.get('headnode') 
                print('.', end ="")
                time.sleep(5)
                if mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1])._info.status== 'FAILED':
                    print("Cluster startup failed, check azure ml run")
                    return None
            self.headnode_private_ip= headnode_private_ip


        params = {}
        max_retry = 20
        count =0
        while count<max_retry:
            params = mlflow.get_run(run_id=self.cluster_job.id.split("/")[-1]).data.params
            count+=1
            time.sleep(1)
            if params !={}:
                break
        if params!={}:
            params.pop("headnode","")
            params.pop("master_ip", "")
            self.mount_points = params
            if headnode_private_ip:
                print("\n cluster is ready, head node ip ",headnode_private_ip)
            else:
                print("\n Cluster started successfully")
        else:
            print("Cluster startup failed, check azure ml run")
            return None

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


    
