#!/bin/bash
#SBATCH --job-name=JOB_NAME		# create a short name for your job
#SBATCH --nodes=1                  	# node count
#SBATCH --ntasks=4               	# number of tasks across all nodes
#SBATCH --mem=64G            		# memory per CPU
#SBATCH --partition=gpuq           	# queue
#SBATCH --gres=gpu:1              	# number of GPUs
#SBATCH --error=error_.%j.err		# error log file name
#SBATCH --output=output_%j.log		# output log file name
#SBATCH --time=6:00:00            	# total run time limit (HH:MM:SS), default is 6 hours
#SBATCH --mail-type=all            	# send email on job start, end and fail

source $HOME/.bashrc
conda activate nobias
python /facility/imganfac/neurogenomics/Testa/Claudio/scripts/analysis_recursive.py






