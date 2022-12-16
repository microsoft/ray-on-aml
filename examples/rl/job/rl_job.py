# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from ray_on_aml.core import Ray_On_AML
import time
import mlflow

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

if __name__ == "__main__":
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray: #in the headnode
        print("head node detected")
        ray.init(address="auto")
        print(ray.cluster_resources())
        algo = (
            PPOConfig()
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(env="CartPole-v1")
            .build()
        )
        for i in range(10):
            result = algo.train()
            print(pretty_print(result))

            if i % 5 == 0:
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")


        



    else:
        print("in worker node")
