from submitit import AutoExecutor
from zencfg import make_config_from_cli

from config.ac_config import ACDefault
from wasp.training import Trainer


def main():
    config = make_config_from_cli(ACDefault)
    config = config.to_dict()

    if config["submit"]:
        slurm_config = config["slurm"]
        executor = AutoExecutor(folder="./logs/%j")
        executor.update_parameters(
            name=config["wandb"]["name"],
            timeout_min=slurm_config["timeout_hours"] * 60,
            nodes=slurm_config["nodes"],
            tasks_per_node=1,
            cpus_per_task=slurm_config["cpus"],
            mem_gb=slurm_config["mem"],
            slurm_partition=slurm_config["partition"],
            slurm_gres=slurm_config["gres"],
            slurm_additional_parameters=slurm_config["additional_parameters"],
            slurm_setup=slurm_config["setup"],
        )
        job = executor.submit(Trainer(), config)
        print(f"Submitted job: {job.job_id}")
    else:
        print("Running locally...")
        Trainer()(config)


if __name__ == "__main__":
    main()
