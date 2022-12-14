import sys
import os

import argparse
import os
import logging
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

#demonstrate pa
if __name__ == "__main__":

    args = parse_args()
    master_ip = args.master_ip
    print("master ip is", master_ip)
    for k,v in args.__dict__.items():
        print(k, v)
