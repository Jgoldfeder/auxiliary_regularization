#!/bin/bash
 
# for this script:
# device seed level

# device dataset num_classes experiment seed val_split level

bash pretrain_script_level5.sh  $1 "torch/cifar100" 100 "exp_cifar100_level5" $2 "validation" $3

bash pretrain_script_level5.sh  $1 "tfds/cars196" 196 "exp_cars196_level5" $2 "test" $3

bash pretrain_script_level5.sh  $1 "tfds/oxford_iiit_pet" 37 "exp_oxford_iiit_pet_level5" $2 "test" $3

bash pretrain_script_level5.sh  $1 "tfds/food101" 101 "exp_food101_level5" $2 "validation" $3

bash pretrain_script_level5.sh  $1 "tfds/stanford_dogs" 120 "exp_stanford_dogs_level5" $2 "test" $3

bash pretrain_script_level5.sh  $1 "tfds/oxford_flowers102" 102 "exp_oxford_flowers102_level5" $2 "validation" $3

bash pretrain_script_level5.sh  $1 "tfds/caltech_birds2011" 200 "exp_caltech_birds2011_level5" $2 "test" $3

bash pretrain_script_level5.sh  $1 "aircraft" 100 "exp_aircraft_level5" $2 "validation" $3 # seems that we only have 100, but site says 102

bash pretrain_script_level5.sh  $1 "tfds/caltech101" 102 "exp_caltech101_level5" $2 "test" $3
