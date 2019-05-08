#!/bin/bash
#
#SBATCH --output=slurm_train_logs/slurm_train_%j.out
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
read -r model_name hyperparams_path <<< $params

python main.py -m $model_name train --training-kind new --hyperparams-path $hyperparams_path
