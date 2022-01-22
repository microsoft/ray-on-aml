import os
import time
import subprocess
import socket
import sys, uuid
import platform
import ray
import inspect
from textwrap import dedent
from azureml.core import  Experiment, Environment, ScriptRunConfig, Run
from azureml.core.runconfig import DockerConfiguration,RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
import logging


__version__='0.0.7'


class Ray_On_AML():
    def __init__(self,compute_cluster=None, ws=None, base_conda_dep =['adlfs==2021.10.0','pip==21.3.1'], 
    base_pip_dep = ['ray[tune]==1.9.2','ray[rllib]==1.9.2','ray[serve]==1.9.2', 'xgboost_ray==0.1.6', 'dask==2021.12.0','pyarrow >= 5.0.0','fsspec==2021.10.1','fastparquet==0.7.2','tabulate==0.8.9'], 
    vnet_rg = None, vm_size=None, vnet=None, subnet=None,
    exp_name ='ray_on_aml', maxnode =5, additional_conda_packages=[],additional_pip_packages=[], job_timeout=600000):
        """ Class for Ray_On_AML
        Ray_On_AML can help you to minimize your effort for configuring Ray Environment to execute data processing and science tasks on Azure Machine Learning Services.

        Example (AML Compute Instance) : 
        >>> from ray_on_aml.core import Ray_On_AML
        >>> ws = Workspace.from_config()
        >>> ray_on_aml =Ray_On_AML(ws=ws,
        >>>                     exp_name='ray_on_aml',
        >>>                     compute_cluster ="cpu-ray-cluster",
        >>>                     additional_pip_packages=['torch==1.10.0', 'torchvision', 'sklearn'],
        >>>                     maxnode=2)

        Currently this moudle will install following packages for Ray on AML.
            Conda
            *    'adlfs==2021.10.0'
            *    'pip==21.3.1'
            
            Pip
            *    'ray[tune]==1.9.2'
            *    'ray[rllib]==1.9.2'
            *    'ray[serve]==1.9.2'
            *    'xgboost_ray==0.1.6'
            *    'dask==2021.12.0'
            *    'pyarrow >= 5.0.0'
            *    'fsspec==2021.10.1'
            *    'fastparquet==0.7.2'
            *    'tabulate==0.8.9'

        This module is compatable with Python 3.8.

        Parameters
        ----------
        ws : Workspace
            The Azure Machine Learning Workspace object.
        computer_cluster : string, (optional), default=None
        base_conda_dep : list, default=['adlfs==2021.10.0','pip==21.3.1']

        base_pip_dep : list, default=['ray[tune]==1.9.2','ray[rllib]==1.9.2','ray[serve]==1.9.2', 'xgboost_ray==0.1.6', 'dask==2021.12.0','pyarrow >= 5.0.0','fsspec==2021.10.1','fastparquet==0.7.2','tabulate==0.8.9']

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

        maxnode : int, (optional), default=5
            You can't change the max number of nodes with this parameter once you created Compute Cluter before.

        additional_conda_packages : list, (optional)
            You can add more package by providing name of packages in a list.         

        additional_pip_packages : list, (optional)
            You can add more package by providing name of packages in a list.

        job_timeout : int, (optional), default=600000

        """
        self.ws = ws
        self.base_conda_dep=base_conda_dep
        self.base_pip_dep= base_pip_dep
        self.vnet_rg=vnet_rg
        self.compute_cluster=compute_cluster
        self.vm_size=vm_size
        self.vnet=vnet
        self.subnet =subnet
        self.exp_name= exp_name
        self.maxnode=maxnode
        self.additional_conda_packages=additional_conda_packages
        self.additional_pip_packages=additional_pip_packages
        self.job_timeout = job_timeout
        # mlflow.set_tracking_uri(self.ws.get_mlflow_tracking_uri())
        # mlflow.set_experiment(self.exp_name)
        # mlflowrun = mlflow.get_run(run.id)


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


    def startRayMaster(self):
        conda_env_name = sys.executable.split('/')[-3]
        logging.info(f"Using {conda_env_name} for the master node")
        #set the the python to this conda env

        cmd =f'. /anaconda/etc/profile.d/conda.sh && conda activate {conda_env_name} && ray stop && ray start --head --port=6379'
        try:
            # if this is not the default environment, it will run
            subprocess.check_output(cmd, shell=True)
        except:
            # User runs this in default environment, just go ahead without activating    
            cmd ='ray stop && ray start --head --port=6379'
            subprocess.check_output(cmd, shell=True)
        ip = self.get_ip()
        return ip


    def checkNodeType(self):
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        if rank is None:
            return "interactive" # This is interactive scenario
        elif rank == '0':
            return "head"
        else:
            return "worker"


    #check if the current node is headnode
    def startRay(self,master_ip=None):
        ip = self.get_ip()
        logging.info("- env: MASTER_ADDR: ", os.environ.get("AZ_BATCHAI_MPI_MASTER_NODE"))
        logging.info("- env: RANK: ", os.environ.get("OMPI_COMM_WORLD_RANK"))
        logging.info("- env: LOCAL_RANK: ", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
        logging.info("- env: NODE_RANK: ", os.environ.get("OMPI_COMM_WORLD_NODE_RANK"))
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        if master_ip is None:
            master_ip = os.environ.get("AZ_BATCHAI_MPI_MASTER_NODE")
        logging.info("- my rank is ", rank)
        logging.info("- my ip is ", ip)
        logging.info("- master is ", master_ip)
        if not os.path.exists("logs"):
            os.makedirs("logs")

        logging.info("free disk space on /tmp")
        os.system(f"df -P /tmp")

        cmd = f"ray start --address={master_ip}:6379"

        logging.info(cmd)

        worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")

        worker_proc = subprocess.Popen(
        cmd.split(),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        )
        self.flush(worker_proc, worker_log)

    def getRay(self, logging_level=logging.ERROR, ci_is_head=True, shm_size='48gb'):
        """This method automatically creates Azure Machine Learning Compute Cluster with Ray, Dask on Ray, Ray Tune, Ray rrlib, and Ray serve.
        This class takes care of all from infrastructure to runtime preperation, it may take 10 mintues for the first time execution of the module.
        Before you run this method, make sure you have existing Virtual Network and subnet in the same Resource Group where Azure Machine Learning Service is.
        If the Virtual Network is not in the same Resource Group then specify the name of Virtual Network, Subnet name.
        This method can also be used in AML job to turn the remote cluster into Ray cluster.
        
        Example (Interactive use with AML Compute Instance) : 
        >>> from ray_on_aml.core import Ray_On_AML
        >>> ws = Workspace.from_config()
        >>> ray_on_aml =Ray_On_AML(ws=ws,
        >>>                     exp_name='ray_on_aml',
        >>>                     compute_cluster ="cpu-ray-cluster",
        >>>                     additional_pip_packages=['torch==1.10.0', 'torchvision', 'sklearn'],
        >>>                     maxnode=2)
        >>> ray = ray_on_aml.getRay()

        Example (Interactive use with AML Compute Instance) : 
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
        shm_size : str, default='48gb'
            Allow the docker container Ray runs in to make full use of the shared memory available from the host OS. Only applicable for interactive use case

        Return
        ----------
            Returns an object of Ray.        

        """
        if self.checkNodeType()=="interactive" and (self.ws is None or self.compute_cluster is None):
            #Interactive scenario, workspace object is require
            raise Exception("For interactive use, please pass AML workspace and compute cluster name to the init")

        if self.checkNodeType()=="interactive":
            return self.getRayInteractive(logging_level,ci_is_head, shm_size)
        elif self.checkNodeType() =='head':
            logging.info("head node detected")
            self.startRayMaster()
            time.sleep(10) # wait for the worker nodes to start first
            ray.init(address="auto", dashboard_port =5000,ignore_reinit_error=True)
            return ray
        else:
            logging.info("workder node detected")
            self.startRay()



    def getRayEnvironment(self):
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

        if Environment.list(self.ws).get(envName) != None:
            return Environment.get(self.ws, envName)
        else:
            python_version = ["python="+platform.python_version()]
            conda_packages = python_version+self.additional_conda_packages +self.base_conda_dep
            pip_packages = self.base_pip_dep +self.additional_pip_packages
            
            print(f"Creating new Environment {envName}")
            rayEnv = Environment(name=envName)

            conda_dep = CondaDependencies()

            for conda_package in conda_packages:
                conda_dep.add_conda_package(conda_package)

            for pip_package in pip_packages:
                conda_dep.add_pip_package(pip_package)

            # Adds dependencies to PythonSection of myenv
            rayEnv.python.conda_dependencies=conda_dep
            rayEnv.register(self.ws)
            
            return rayEnv


    def getRayInteractive(self, logging_level, ci_is_head, shm_size):
        """Create Compute Cluster, an entry script and Environment

        Create Compute Cluster if given name of Compute Cluster doesn't exist in Azure Machine Learning Workspace

        Get Azure Environement for Ray runtime

        Generate entry script to run Ray in Compute Cluster

        If the script run on Compute Cluster successfully, Ray object will be returned.

        """
        # Verify that cluster does not exist already
        self.shutdown(end_all_runs=True)
        ws_detail = self.ws.get_details()
        ws_rg = ws_detail['id'].split("/")[4]

        try:
            ray_cluster = ComputeTarget(workspace=self.ws, name=self.compute_cluster)
            print(f'Found existing cluster {self.compute_cluster}')
        except ComputeTargetException:
            if self.vnet_rg is None:
                vnet_rg = ws_rg
            else:
                vnet_rg = self.vnet_rg
            
            compute_config = AmlCompute.provisioning_configuration(vm_size=self.vm_size,
                                                                min_nodes=0, max_nodes=self.maxnode,
                                                                vnet_resourcegroup_name=vnet_rg,
                                                                vnet_name=self.vnet,
                                                                subnet_name=self.subnet)
                                                                
            ray_cluster = ComputeTarget.create(self.ws, self.compute_cluster, compute_config)

            ray_cluster.wait_for_completion(show_output=True)

        rayEnv = self.getRayEnvironment()

        ##Create the source file
        os.makedirs(".tmp", exist_ok=True)



        source_file_content = """
        import os
        import time
        import subprocess
        import threading
        import socket
        import sys, uuid
        import platform
        from azureml.core import Run
        import ray
        run = Run.get_context()
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
        
            cmd ='ray start --head --port=6379'
            subprocess.Popen(
            cmd.split(),
            universal_newlines=True
            )
            ip = socket.gethostbyname(socket.gethostname())
            run.log("headnode", ip)
            time.sleep({0})


        def checkNodeType():
            rank = os.environ.get("RANK")
            if rank is None:
                return "interactive" # This is interactive scenario
            elif rank == '0':
                return "head"
            else:
                return "worker"


        def startRay(master_ip=None):
            ip = socket.gethostbyname(socket.gethostname())
            print("- env: MASTER_ADDR: ", os.environ.get("AZ_BATCHAI_MPI_MASTER_NODE"))
            print("- env: RANK: ", os.environ.get("OMPI_COMM_WORLD_RANK"))
            print("- env: LOCAL_RANK: ", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
            print("- env: NODE_RANK: ", os.environ.get("OMPI_COMM_WORLD_NODE_RANK"))
            rank = os.environ.get("OMPI_COMM_WORLD_RANK")

            master = os.environ.get("AZ_BATCHAI_MPI_MASTER_NODE")
            print("- my rank is ", rank)
            print("- my ip is ", ip)
            print("- master is ", master)
            if not os.path.exists("logs"):
                os.makedirs("logs")

            print("free disk space on /tmp")
            os.system(f"df -P /tmp")
            if master_ip is None:
                master_ip =os.environ.get("MASTER_ADDR")

            cmd = "ray start --address="+master_ip+":6379"

            print(cmd)

            worker_log = open("logs/worker_"+rank+"_log.txt", "w")

            worker_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            )
            flush(worker_proc, worker_log)

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


        """.format(self.job_timeout)

        source_file = open(".tmp/source_file.py", "w")
        source_file.write(dedent(source_file_content))
        source_file.close()

        if ci_is_head:
            master_ip = self.startRayMaster()
        else:
            master_ip= "None"
        docker_config = DockerConfiguration(use_docker=True, shm_size=shm_size)
        aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
        aml_run_config_ml.target = ray_cluster
        aml_run_config_ml.docker =docker_config
        aml_run_config_ml.node_count = self.maxnode
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
                    return ray
        else:
            print("Waiting cluster to start and return head node ip")
            while not 'headnode' in run.get_metrics("headnode"):
                print('.', end ="")
                time.sleep(5)
                active_run = Run.get(self.ws,run.id)
                if active_run.status == 'Failed':
                    print("Cluster startup failed, check detail at run")
                    return None, None
            headnode_private_ip = run.get_metrics("headnode")['headnode']
            print('Headnode has IP:', headnode_private_ip)
            self.headnode_private_ip= headnode_private_ip
            try:
                #disconnect client first to make sure only one is active at a time
                ray.disconnect()
            except:
                pass
            ray.init(f"ray://{headnode_private_ip}:10001",ignore_reinit_error=True, logging_level=logging_level)
            return ray

    def getRun(self):
        """Return run that associated with
        """
        if Run.get(self.ws, self.run.id) != None:
            self.run = Run.get(self.ws, self.run.id)
            return self.run
        else:
            return None

        
    def shutdown(self, end_all_runs=True):
        """Stop Ray and Compute Cluster

        Parameters
        ----------
        end_all_runs : bool, default=True
            Stop all your compute cluster by default.
            If you want to stop your own Compute Cluster, you can your following
                ray_on_aml.shutdown(end_all_runs=False)
        """
        def end_all_run():
            exp= Experiment(self.ws, self.exp_name)
            runs = exp.get_runs()
            for run in runs:
                if (run.status =='Running') or (run.status =='Preparing'):
                    print("Canceling active run ", run.id, "in", self.exp_name)
                    run.cancel()

        if end_all_runs:
            print("Cancel active AML runs if any")
            end_all_run()
        else:
            print("Cancel your run")
            self.run.cancel()

        try:
            print("Shutting down ray if any")
            self.ray.shutdown()
        except:
            # print("Cannot shutdown ray, ray was not there")
            pass
    
        
