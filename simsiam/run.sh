python main_simsiam.py \
  -a resnet50 \
  --gpu 0 \
  --batch-size 64 \
  --print-freq 100 \
  --seed 5 \
  --dataset-download \
  --dataset aircraft \
  --train-split "test" \

# python generate_labels.py \
#   --gpu 0 \
#   --dataset tfds/caltech101 \
#   --seed 5