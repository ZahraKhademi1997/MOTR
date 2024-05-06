#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=configuration
#SBATCH --mail-type=NONE
#SBATCH --mail-user=
#SBATCH --time=71:30:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:8   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   
#SBATCH --mem-per-cpu=4gb
#SBATCH --constraint=a100


# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)

module load cuda
module load conda/24.1.2 gcc/12.2.0
conda activate seg-training
# conda activate motr-seg


# PRINTS
#=======
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST" 
echo "Other nodes: $NODES"

echo "SLURM Environment Variables:"
env | grep SLURM
echo "---------------------------"

# PYTHON SCRIPT
#==============
echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."
# python -u main.py
python -u models/ops/setup.py build install