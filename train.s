#!/bin/bash
#
#SBATCH --output=slurm_train_%j.out
#SBATCH --job-name=AAI_proj
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:p40:1

module load python3/intel/3.6.3

source /home/dam740/pytorch_venv/bin/activate

line_number=$SLURM_ARRAY_TASK_ID
params=$(sed -n ${line_number}p to_train.txt)
read -r model_name training_kind hyperparams_path <<< $params

python main.py -m $model_name train --training-kind $training_kind --hyperparams-path $hyperparams_path

# test-run server
#python main.py -m cnn --test-run train --training-kind new --hyperparams-path /scratch/dam740/AAI_project/hyperparams/cnn.yaml
#python main.py -m cnn_rnn --test-run train --training-kind new --hyperparams-path /scratch/dam740/AAI_project/hyperparams/cnn_rnn_add.yaml

# local command
# python main.py -m cnn_rnn --test-run --local train --training-kind new --hyperparams-path hyperparams/cnn_rnn_add.yaml
