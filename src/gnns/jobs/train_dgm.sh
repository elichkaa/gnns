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
export USE_KEOPS=True

source /home/ka/ka_stud/ka_ufszm/.cache/pypoetry/virtualenvs/gnns-tZ7_1okr-py3.11/bin/activate

cd ~/project/gnns/src/gnns

srun python train.py \
    --model_type dgm \
    --encoder_name distilbert-base-uncased \
    --max_epochs 30 \
    --batch_size 8 \
    --lr 1e-3 \
    --dropout 0.3 \
    --weight_decay 0.001 \
    --patience 10 \
    --pooling mean \
    --max_length 512 \
    --k 5 \
    --distance euclidean \
    --gfun gat \
    --ffun gcn \
    --test_eval 5 \
    --resume_from_checkpoint "logs/dDGM_distilbert-base-uncased_k5_gat_euclidean_poolmean/version_0/checkpoints/epoch=14-step=16980.ckpt" \
    --freeze_encoder