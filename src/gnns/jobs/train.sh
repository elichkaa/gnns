#!/bin/bash
#SBATCH --job-name=graph_gnn
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=127500mb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export USE_KEOPS=False
export TORCH_USE_CUDA_DSA=True

source /home/ka/ka_stud/ka_ufszm/.cache/pypoetry/virtualenvs/gnns-tZ7_1okr-py3.11/bin/activate

# srun python ../train.py \
#     --model_type baseline \
#     --encoder_name distilbert-base-uncased \
#     --max_epochs 30 \
#     --batch_size 32 \
#     --lr 1e-3 \
#     --dropout 0.3 \
#     --weight_decay 1e-4 \
#     --patience 5 \
#     --pooling mean \
#     --max_length 512 \
#     --freeze_encoder

srun python ../train.py \
    --model_type baseline \
    --encoder_name distilbert-base-uncased \
    --max_epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --patience 5 \
    --pooling mean \
    --max_length 512 \
    --freeze_encoder