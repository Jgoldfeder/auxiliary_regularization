#!/bin/bash

# device dataset num_classes experiment seed val_split

# generate contrastive labels
python simsiam/main_simsiam.py -a resnet50 --gpu $1 --dataset $2 --batch-size 64 --print-freq 500 --seed $5 --dataset-download

python simsiam/generate_labels.py -a resnet50 --gpu $1 --dataset-download --dataset $2 --num-classes $3 --batch-size 64 --seed $5

# run experiments
CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 baseline seed $5 lr=2e-4" --log-wandb --seed $5 --val-split $6

CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 metabalance 1 1 seed $5 lr=2e-4" --log-wandb --dual --dual-weights 1 1 --seed $5 --metabalance --val-split $6

CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 contrastive 1 1 seed $5 lr=2e-4" --log-wandb --dual --dual-weights 1 1 --seed $5 --metabalance --contrastive --val-split $6

CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-3 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 baseline seed $5 lr=2e-3" --log-wandb --seed $5 --val-split $6

CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-3 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 metabalance 1 1 seed $5 lr=2e-3" --log-wandb --dual --dual-weights 1 1 --seed $5 --metabalance --val-split $6

CUDA_VISIBLE_DEVICES=$1 python train.py $2 -d $2  --num-classes=$3 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-3 --model=resnet50 --opt=adam --weight-decay=1e-4 --experiment "$4" --name "resnet50 contrastive 1 1 seed $5 lr=2e-3" --log-wandb --dual --dual-weights 1 1 --seed $5 --metabalance --contrastive --val-split $6