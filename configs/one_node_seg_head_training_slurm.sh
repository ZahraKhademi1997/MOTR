#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=segHead_cross_AA
#SBATCH --mail-type=NONE
#SBATCH --mail-user=
#SBATCH --time=71:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --gpus-per-node=a100:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8gb
#SBATCH --constraint=a100




# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)

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
python -m torch.distributed.launch --nproc_per_node=6 \
    --use_env main.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --masks \
    --epoch 500 \
    --with_box_refine \
    --lr_drop 200 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_PerPixelEmbedding 4e-5 \
    --lr_AxialBlock 2e-4 \
    --lr_seg_branches 2e-7 \
    --lr_pos_cross_attention 2e-7 \
    --pretrained /blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-segmentation-head-training/outputs/pretrained_checkpoints/checkpoint0299.pth \
    --output_dir /blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-segmentation-head-training/outputs/modek_checkpoints_200queries_AppleMOTS/ \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_steps 50 90 150 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --mot_path /blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-segmentation-head-training/data/Dataset/ \
    --exp_name pub_submit_15 \