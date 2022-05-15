# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
from src.ray_on_aml.core import Ray_On_AML
import time
from azureml.core import Run



def on_train_result(info):
    '''Callback on train result to record metrics returned by trainer.
    '''
    run = Run.get_context()
    run.log(
        name='episode_reward_mean',
        value=info["result"]["episode_reward_mean"])
    run.log(
        name='episodes_total',
        value=info["result"]["episodes_total"])

def train():
# Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "Taxi-v3",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
        
        "callbacks": {"on_train_result": on_train_result},

    }

    # Create our RLlib Trainer.
    trainer = PPOTrainer(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(3):
        print(trainer.train())

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
#     trainer.evaluate()



if __name__ == "__main__":
    ray_on_aml =Ray_On_AML()
    ray = ray_on_aml.getRay()

    for item, value in os.environ.items():
        print('{}: {}'.format(item, value))

    if ray: #in the headnode
        print("head node detected")
        time.sleep(15)
        print(ray.cluster_resources())
        train()



    else:
        print("in worker node")
