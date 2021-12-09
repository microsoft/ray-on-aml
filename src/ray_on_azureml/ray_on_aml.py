import os
import time
import subprocess
import threading
import socket
import sys, uuid
import platform
import mlflow
import ray
import inspect
from textwrap import dedent
from azureml.core import Workspace, Experiment, Environment, Datastore, Dataset, ScriptRunConfig, Run
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
class Ray_On_AML():
# base_conda_dep =['gcsfs','fs-gcsfs','numpy','h5py','scipy','toolz','bokeh','dask','distributed','matplotlib','pandas','pandas-datareader','pytables','snakeviz','ujson','graphviz','fastparquet','dask-ml','adlfs','pytorch','torchvision','pip'], base_pip_dep = ['azureml-defaults','python-snappy', 'fastparquet', 'azureml-mlflow', 'ray[default]==1.8.0', 'xgboost_ray', 'raydp', 'xgboost', 'pyarrow==4.0.1']
    
    def __init__(self, ws, base_conda_dep =['adlfs','pytorch','matplotlib','torchvision','pip'], base_pip_dep = ['sklearn','xgboost','lightgbm','ray[tune]==1.9.0', 'xgboost_ray', 'dask'], vnet_rg = None, compute_cluster = 'ray_on_aml', vm_size='STANDARD_DS3_V2',vnet='rayvnet', subnet='default', exp ='ray_on_aml', maxnode =5, additional_conda_packages=[],additional_pip_packages=[], job_timeout=60000):
        self.ws = ws
        self.base_conda_dep=base_conda_dep
        self.base_pip_dep= base_pip_dep
        self.vnet_rg=vnet_rg
        self.compute_cluster=compute_cluster
        self.vm_size=vm_size
        self.vnet=vnet
        self.subnet =subnet
        self.exp= exp
        self.maxnode=maxnode
        self.additional_conda_packages=additional_conda_packages
        self.additional_pip_packages=additional_pip_packages
        self.job_timeout = job_timeout
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
        print(conda_env_name)
        #set the the python to this conda env

        cmd =f'. $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate {conda_env_name} && ray stop && ray start --head --port=6379 --object-manager-port=8076'
        try:
            #if this is not the default environment, it will run
            subprocess.check_output(cmd, shell=True)
        except:
    #         User runs this in default environment, just goahead without activating
            cmd ='ray stop && ray start --head --port=6379 --object-manager-port=8076'
            subprocess.check_output(cmd, shell=True)
        ip = self.get_ip()
        return ip
    def startRay(self,master_ip=None):
        ip = get_ip()
        print("- env: MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
        print("- env: MASTER_PORT: ", os.environ.get("MASTER_PORT"))
        print("- env: RANK: ", os.environ.get("RANK"))
        print("- env: LOCAL_RANK: ", os.environ.get("LOCAL_RANK"))
        print("- env: NODE_RANK: ", os.environ.get("NODE_RANK"))
        rank = os.environ.get("RANK")

        master = os.environ.get("MASTER_ADDR")
        print("- my rank is ", rank)
        print("- my ip is ", ip)
        print("- master is ", master)
        if not os.path.exists("logs"):
            os.makedirs("logs")

        print("free disk space on /tmp")
        os.system(f"df -P /tmp")

        cmd = f"ray start --address={master_ip}:6379 --object-manager-port=8076"

        print(cmd)

        worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")

        worker_proc = subprocess.Popen(
        cmd.split(),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        )
        self.flush(worker_proc, worker_log)
        time.sleep(self.job_timeout)


    def getRay(self):
        #create or get the cluster with maxnode, then create an experiment to create ray cluster, get current ray and create ray init and return ray object
    #     try:
    #         #logic to check if ray cluster is already started and running, then return ray

    #     except:
            #ray is not there, go ahead and create
        #start current node master
        master_ip = self.startRayMaster()
        # Verify that cluster does not exist already
        ws_detail = self.ws.get_details()
        ws_rg = ws_detail['id'].split("/")[4]
        try:
            ray_cluster = ComputeTarget(workspace=self.ws, name=self.compute_cluster)
            print('Found existing cluster, use it.')
        except ComputeTargetException:
            if vnet_rg is None:
                vnet_rg = self.ws.ws_rg
            compute_config = AmlCompute.provisioning_configuration(vm_size=self.vm_size,
                                                                min_nodes=0, max_nodes=self.maxnode,
                                                                vnet_resourcegroup_name=vnet_rg,
                                                                vnet_name=self.vnet,
                                                                subnet_name=self.subnet)
            ray_cluster = ComputeTarget.create(self.ws, self.compute_cluster, compute_config)

            ray_cluster.wait_for_completion(show_output=True)


        python_version = ["python="+platform.python_version()]



        conda_packages = python_version+self.additional_conda_packages +self.base_conda_dep
        pip_packages = self.base_pip_dep +self.additional_pip_packages

        rayEnv = Environment(name="rayEnv")
        dockerfile = r"""
        FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04
        ARG HTTP_PROXY
        ARG HTTPS_PROXY

        # set http_proxy & https_proxy
        ENV http_proxy=${HTTP_PROXY}
        ENV https_proxy=${HTTPS_PROXY}

        RUN http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} apt-get update -y \
            && mkdir -p /usr/share/man/man1 \
            && http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} apt-get install -y openjdk-8-jdk \
            && mkdir /raydp \
            && pip --no-cache-dir install raydp

        WORKDIR /raydp

        # unset http_proxy & https_proxy
        ENV http_proxy=
        ENV https_proxy=

        """
#         dockerfile = r"""
#         FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
#         # Install OpenJDK-8
#         RUN apt-get update -y \
#         && mkdir -p /usr/share/man/man1 \
#         && apt-get install -y openjdk-8-jdk 

#         """

        # Set the base image to None, because the image is defined by Dockerfile.
        rayEnv.docker.base_image = None
        rayEnv.docker.base_dockerfile = dockerfile

        conda_dep = CondaDependencies()

        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        for pip_package in pip_packages:
            conda_dep.add_pip_package(pip_package)

        # Adds dependencies to PythonSection of myenv
        rayEnv.python.conda_dependencies=conda_dep

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
        #import mlflow
        import ray
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

        def startRay(master_ip=None):
            ip = socket.gethostbyname(socket.gethostname())
            print("- env: MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
            print("- env: MASTER_PORT: ", os.environ.get("MASTER_PORT"))
            print("- env: RANK: ", os.environ.get("RANK"))
            print("- env: LOCAL_RANK: ", os.environ.get("LOCAL_RANK"))
            print("- env: NODE_RANK: ", os.environ.get("NODE_RANK"))
            rank = os.environ.get("RANK")

            master = os.environ.get("MASTER_ADDR")
            print("- my rank is ", rank)
            print("- my ip is ", ip)
            print("- master is ", master)
            if not os.path.exists("logs"):
                os.makedirs("logs")

            print("free disk space on /tmp")
            os.system(f"df -P /tmp")
            cmd = f"ray start --address={master_ip}:6379 --object-manager-port=8076"
            worker_log = open("logs/worker_{rank}_log.txt".format(rank=rank), "w")

            worker_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            )
            flush(worker_proc, worker_log)

            time.sleep(60000)

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--master_ip")
            args, unparsed = parser.parse_known_args()
            master_ip = args.master_ip
            startRay(master_ip)

        """


        source_file = open(".tmp/source_file.py", "w")
        n = source_file.write(dedent(source_file_content))
        source_file.close()
        src = ScriptRunConfig(source_directory='.tmp',
                           script='source_file.py',
                           environment=rayEnv,
                           compute_target=ray_cluster,
                           distributed_job_config=PyTorchConfiguration(node_count=self.maxnode),
                              arguments = ["--master_ip",master_ip]
                           )
        run = Experiment(self.ws, self.exp).submit(src)
        time.sleep(10)
        ray.shutdown()
        ray.init(address="auto",ignore_reinit_error=True)
        self.run = run
        self.ray = ray
        while True:
            active_run = Run.get(self.ws,run.id)
            if active_run.status != 'Running':
                print("Waiting: Cluster status is in ", active_run.status)
                time.sleep(10)
            else:
                return active_run, ray
        
    def shutdown(self):
        try:
            self.run.cancel()
        except:
            print("Run does not exisit, finding active run to cancel")
            exp= Experiment(self.ws,self.exp)
            runs = exp.get_runs()
            for run in runs:
                if run.status =='Running':
                    print("Get active run ", run.id)
                    run.cancel()
        try:
            self.ray.shutdown()
        except:
            print("Cannot shutdown ray")
    
        
