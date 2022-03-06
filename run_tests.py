import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="config", config_name="config")
def run_test(cfg: DictConfig):
    main_dir = hydra.utils.get_original_cwd()
    create_slurm(main_dir)


def create_slurm(main_path):
    f = open("menakbert_job.slurm", "a")
    f.write("#! /bin/sh\n")
    f.write("#SBATCH --job-name=menakbert\n")
    f.write("#SBATCH --output=menakbert.out\n")
    f.write("#SBATCH --error=menakbert.err\n")
    f.write("#SBATCH --partition=studentbatch\n")
    f.write("#SBATCH --time=1\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write("#SBATCH --gpus=1\n")
    f.write(f"python {main_path}main.py")
    f.close()
    os.system("sbatch menakbert_job.slurm")

if __name__ == '__main__':
    run_test()