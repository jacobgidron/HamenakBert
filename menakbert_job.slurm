#! /bin/sh
#SBATCH --job-name=menakbert
#SBATCH --output=menakbert.out
#SBATCH --error=menakbert.err
#SBATCH --partition=studentbatch
#SBATCH --time=2
#SBATCH --nodes=1
#SBATCH --nodelist=s-006
#SBATCH --ntasks=1
#SBATCH --gpus=1
python run_tests.py -m base_path=/home/joberant/NLP_2122/pinto/menakbert hyper_params.train_batch_size=100 hyper_params.max_epochs=25 hyper_params.min_epochs=5 hyper_params.dropout=0,0.1,0.2,0.3,0.4,0.5 hyper_params.lr=0.000002,0.0000015,0.000001,0.0000005