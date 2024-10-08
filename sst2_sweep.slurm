#!/bin/bash
 
#SBATCH --job-name=sst2_sweep_8_T1000
#SBATCH --output=<name>-%A.out
#SBATCH --error=<name>-%A.err
#SBATCH --mail-user= TODO REPLACE
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
 
# Load python module
ml python/anaconda3
 
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate
 
# If the following does not work, try 'source activate <env-name>'
conda activate augmult
 
# Cache data to local /tmp directory (optional)
#rsync -r /vol/aimspace/projects/<dataset> /tmp
 
# Run the program
python execute_sweep.py sst2/4y5dpnvd