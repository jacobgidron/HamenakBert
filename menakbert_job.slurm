#! /bin/sh
#SBATCH --job-name=menakbert
#SBATCH --output=menakbert.out
#SBATCH --error=menakbert.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python main.py hyper_params.lr=1,2 hyper_params.dropout=0.1,0.2
