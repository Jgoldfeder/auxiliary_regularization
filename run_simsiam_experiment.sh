python simsiam/main_simsiam.py \
  -a resnet50 \
  --gpu 1 \
  --batch-size 64 \
  --print-freq 100 \
  --seed 5 \
  --dataset-download \
  --dataset tfds/oxford_iiit_pet \
  --train-split "test" \

python simsiam/generate_labels.py \
  -a resnet50 \
  --gpu 1 \
  --dataset-download \
  --num-classes 37 \
  --dataset tfds/oxford_iiit_pet \
  --batch-size 64 \
  --seed 5