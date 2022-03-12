import csv

import hydra
from omegaconf import DictConfig
import os

CSV_HEAD = [
    "train_data",
    "val_data",
    "test_data",
    "model",
    "maxlen",
    "minlen",
    "lr",
    "dropout",
    "train_batch_size",
    "val_batch_size",
    "max_epochs",
    "min_epochs",
    "weighted_loss",
    "path",
    "acc_S",
    "acc_D",
    "acc_N",
]

@hydra.main(config_path="config", config_name="config")
def run_test(cfg: DictConfig):

    with open("result_tabel.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEAD)
    main_dir = hydra.utils.get_original_cwd()
    create_slurm(main_dir,cfg.max_epochs)


def create_slurm(main_path, max_epochs):
    f = open("menakbert_job.slurm", "a")
    f.write("#! /bin/sh\n")
    f.write("#SBATCH --job-name=menakbert\n")
    f.write("#SBATCH --output=menakbert.out\n")
    f.write("#SBATCH --error=menakbert.err\n")
    f.write("#SBATCH --partition=studentbatch\n")
    f.write(f"#SBATCH --time={3 * max_epochs}\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write("#SBATCH --gpus=1\n")
    f.write(f"python {main_path}/main.py base_path={main_path}")
    f.close()
    os.system("sbatch menakbert_job.slurm")


if __name__ == '__main__':
    run_test()
