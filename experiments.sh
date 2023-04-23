#!/bin/bash
 
# for this script:
# device seed

# device dataset num_classes experiment seed val_split 

bash pretrain_script.sh  $1 "torch/cifar100" 100 "exp_cifar100" $2 "validation"

bash pretrain_script.sh  $1 "tfds/cars196" 196 "exp_cars196" $2 "test"

bash pretrain_script.sh  $1 "tfds/oxford_iiit_pet" 37 "exp_oxford_iiit_pet" $2 "test"

bash pretrain_script.sh  $1 "tfds/food101" 101 "exp_food101" $2 "validation"

bash pretrain_script.sh  $1 "tfds/stanford_dogs" 120 "exp_stanford_dogs" $2 "test"

bash pretrain_script.sh  $1 "tfds/oxford_flowers102" 102 "exp_oxford_flowers102" $2 "validation"

bash pretrain_script.sh  $1 "tfds/caltech_birds2011" 200 "exp_caltech_birds2011" $2 "test"

bash pretrain_script.sh  $1 "aircraft" 100 "exp_aircraft" $2 "validation" # seems that we only have 100, but site says 102

bash pretrain_script.sh  $1 "tfds/caltech101" 102 "exp_caltech101" $2 "test"
