CUDA_VISIBLE_DEVICES=2 python3 train.py torch/cifar100 -d torch/cifar100  --num-classes=100 -b=64 --img-size=224 --epochs=50 --color-jitter=0   --sched='cosine' --model-ema --model-ema-decay=0.995  --reprob=0.5 --smoothing=0.1  --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4 --model=resnet50 --opt=adam --weight-decay=1e-4 --output data/auxiliary/contrastive --log-interval 150 --log-wandb --dataset-download
