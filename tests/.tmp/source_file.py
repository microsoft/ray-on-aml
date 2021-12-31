
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

    cmd ='ray start --head --port=6379 --object-manager-port=8076'
    subprocess.Popen(
    cmd.split(),
    universal_newlines=True
    )
    ip = socket.gethostbyname(socket.gethostname())
    run.log("headnode", ip)
    time.sleep(6000)


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
    if master_ip is None:
        master_ip =os.environ.get("MASTER_ADDR")

    cmd = f"ray start --address={master_ip}:6379 --object-manager-port=8076"

    print(cmd)

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
    #check if the user wants CI to be headnode
    if master_ip !="None": 
        startRay(master_ip)
    else:
        if checkNodeType() =="head":
            startRayMaster()
        else:
            time.sleep(20)
            startRay()


