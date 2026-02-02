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

# distilbert-base-uncased

srun python train_v2.py \
     --model_type dgm \
     --use_continuous_dgm \
     --encoder_name google/embeddinggemma-300m \
     --dataset mrd \
     --task regression \
     --gfun gcn \
     --k 15 \
     --batch_size 32
     
# --dgm_layers "[[768, 32], [544, 32], [288, 32]]" \
# --conv_layers "[[768, 512], [512, 256], [256, 128]]" \
# --fc_layers "[128, 64, 1]" \
# --pre_fc "[]"

# --fc_layers "[512, 1]" \
#      --dgm_layers "[[768, 512]]" \
#      --conv_layers "[[768, 512]]" \
# srun python train_v2.py \
#     --model_type dgm \
#     --encoder_name google/embeddinggemma-300m \
#     --max_length 256 \
#     --max_epochs 200 \
#     --batch_size 16 \
#     --k 10 \
#     --lr 1e-5 \
#     --dropout 0.3 \
#     --patience 30 \
#     --weight_decay 0.01

# srun python train_v2.py \
#     --model_type dgm \
#     --encoder_name google/embeddinggemma-300m \
#     --max_length 256 \
#     --max_epochs 100 \
#     --batch_size 16 \
#     --lr 1e-5 \
#     --dropout 0.3 \
#     --patience 15 \
#     --weight_decay 0.01

# python train.py \
#     --model_type cdgm \
#     --encoder_name google/embeddinggemma-300m \
#     --max_length 256 \
#     --batch_size 64 \
#     --lr 2e-4 \
#     --dropout 0.2 \
#     --k 10 \
#     --gfun gat \
#     --distance euclidean \
#     --pooling mean \
#     --max_epochs 50 \
#     --patience 20 \
#     --weight_decay 0.01

# srun python train.py \
#     --model_type cdgm \
#     --encoder_name google/embeddinggemma-300m \
#     --max_epochs 30 \
#     --batch_size 32 \
#     --lr 5e-6 \
#     --dropout 0.1 \
#     --weight_decay 1e-5 \
#     --max_length 256 \
#     --resume_from_checkpoint "./logs/cdgm_google-embeddinggemma-300m_k5_gat_euclidean_poolmean/version_14/checkpoints/epoch=14-step=2130.ckpt"

# srun python train.py \
#     --model_type dgm \
#     --encoder_name google/embeddinggemma-300m \
#     --max_epochs 30 \
#     --batch_size 16 \
#     --lr 5e-4 \
#     --dropout 0.4 \
#     --weight_decay 0.01 \
#     --patience 10 \
#     --pooling mean \
#     --max_length 512 \
#     --k 30 \
#     --distance euclidean \
#     --gfun gat \
#     --ffun gcn \
#     --test_eval 10 \
#     --dgm_layers "[[768, 512]]" \
#     --conv_layers "[[768, 512]]" \
#     --fc_layers "[512, 256, 128, 20]" \
#     --lambda_sparse 0.1 \
#     --lambda_entropy 0.05 \
    # --resume_from_checkpoint "logs/dDGM_google-embeddinggemma-300m_k15_gat_euclidean_poolmean/version_24/checkpoints/epoch=29-step=16980.ckpt"