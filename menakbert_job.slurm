#! /bin/sh
#SBATCH --job-name=menakbert
#SBATCH --output=menakbert.out
#SBATCH --error=menakbert.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --nodelist=s-006
#SBATCH --ntasks=1
#SBATCH --gpus=1
python run_tests.py -m hyper_params.lr=0.000002,0.0000015,0.000001,0.0000005 hyper_params.dropout=0,0.1,0.2,0.3,0.4,0.5 hyper_params.weighted_loss=True,False base_path=/home/joberant/NLP_2122/pinto/menakbert
