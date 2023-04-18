python main_simsiam.py \
  -a resnet50 \
  --gpu 0 \
  --batch-size 64 \
  --print-freq 100 \
  --dim 4096 \
  --pred-dim 1024 \
  cifar100

python generate_labels.py \
  --gpu 0 \
  --dim 4096 \
  --pred-dim 1024