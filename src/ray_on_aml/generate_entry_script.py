from textwrap import dedent

def generate_entry_script(job_timeout, master_ip_env_name, world_rank_env_name, ray_start_head_args, ray_start_worker_args):
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
        class _EventLogger:

            @staticmethod
            def get_logger(name):
                logger = logging.getLogger(__name__).getChild(name)
                logger.propagate = False
                logger.setLevel(logging.INFO)
            
                try:
                    from opencensus.ext.azure.log_exporter import AzureEventHandler

                    # Doc: Set up Azure Monitor for your Python application
                    # https://learn.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python#send-events
                    if not _EventLogger._found_handler(logger, AzureEventHandler):
                        logger.addHandler(
                            AzureEventHandler(
                                connection_string="InstrumentationKey=" + instrumentation_key
                            )
                        )
                except ImportError:
                    pass

                return logger
            
            @staticmethod
            def track_event(logger, name, properties=None):
                custom_dimensions = _EventLogger._try_get_run_info()
                if properties is not None:
                    custom_dimensions.update(properties)
                    
                logger.info(name, extra={{"custom_dimensions": custom_dimensions}})

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
            module_event_logger = _EventLogger.get_logger(__name__)
            args = parse_args()
            master_ip = args.master_ip
            #log mount points (inputs, outputs) to the param so that users can use
            for k,v in args.__dict__.items():
                mlflow.log_param(k, v)
             #check if the user wants CI to be headnode
            if master_ip !="None": 
                _EventLogger.track_event(module_event_logger, "getRay",{{"run_mode":"interactive_ci_head"}})
                startRay(master_ip)

            else:
                if checkNodeType() =="head":
                    _EventLogger.track_event(module_event_logger, "getRay",{{"run_mode":"interactive_client"}})
                    startRayMaster()
                else:
                    time.sleep(20)
                    startRay()
        """.format(job_timeout, master_ip_env_name, world_rank_env_name, ray_start_head_args, ray_start_worker_args)
    return dedent(source_file_content)