import logging
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

def get_aml_env_v2(ml_client, envName, conda_dep, base_image):
    try:
        return self.ml_client.environments._get_latest_version(envName)
    except:
        pass

    # If UserError raised
    # Create new Environment
    logging.info(f"Creating new Environment {envName}")
    conda_dep.save(".tmp/conda.yml")

    rayEnv = Environment(
        image=base_image,
        conda_file=".tmp/conda.yml",
        name=envName,
        description="Environment for ray cluster.",
    )

    self.ml_client.environments.create_or_update(rayEnv)
    return 