# python simsiam/main_simsiam.py \
#   -a resnet50 \
#   --gpu 1 \
#   --batch-size 64 \
#   --print-freq 100 \
#   --seed 5 \
#   --dataset-download \
#   --dataset tfds/caltech101 \
#   --train-split "test" \

python simsiam/generate_labels.py \
  --gpu 1 \
  --dataset-download \
  --num-classes 102 \
  --dataset tfds/caltech101 \
  --seed 5