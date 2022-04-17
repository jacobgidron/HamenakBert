#! /bin/sh
#SBATCH --job-name=menakbert
#SBATCH --output=menakbert.out
#SBATCH --error=menakbert.err
#SBATCH --partition=studentbatch
#SBATCH --time=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python run_tests.py -m base_path=/home/joberant/NLP_2122/pinto/menakbert hyper_params.train_batch_size=64 hyper_params.dropout=0.3 hyper_params.lr=1e-5 hyper_params.linear_layer_size=2048,4096 hyper_params.dropout=0.3,0.4