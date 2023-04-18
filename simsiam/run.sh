python main_simsiam.py \
  -a resnet50 \
  --gpu 0 \
  --batch-size 64 \
  --print-freq 100 \
  --seed 4 \
  cifar100

python generate_labels.py \
  --gpu 0 \
  --seed 4

python main_simsiam.py \
  -a resnet50 \
  --gpu 0 \
  --batch-size 64 \
  --print-freq 100 \
  --seed 5 \
  cifar100

python generate_labels.py \
  --gpu 0 \
  --seed 5