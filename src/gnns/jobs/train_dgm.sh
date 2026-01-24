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

srun python ../train.py \
    --model_type dgm \
    --encoder_name google/embeddinggemma-300m \
    --max_epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --dropout 0.5 \
    --weight_decay 0.001
    --patience 7 \
    --pooling mean \
    --max_length 512 \
    --k 10 \
    --distance euclidean \
    --gfun gat \
    --ffun gcn \
    # -- 
    #--conv_layers "[[768, 384], [384, 192], [192, 96]]" \
    #--fc_layers "[96, 48, 20]" \
    --pre_fc "[]" \
    --lambda_sparse 0.2 \
    --lambda_entropy 0.05 \