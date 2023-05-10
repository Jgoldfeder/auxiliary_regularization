#!/bin/bash
 
# for this script:
# device seed level

# device dataset num_classes experiment seed val_split level

bash pretrain_script_arbitraryLevel.sh  $1 "torch/cifar100" 100 "newExp_cifar100_level${3}" $2 "validation" $3

#bash pretrain_script_arbitraryLevel.sh  $1 "tfds/cars196" 196 "newExp_cars196_level${3}" $2 "test" $3

bash pretrain_script_arbitraryLevel.sh  $1 "tfds/oxford_iiit_pet" 37 "newExp_oxford_iiit_pet_level${3}" $2 "test" $3

#bash pretrain_script_arbitraryLevel.sh  $1 "tfds/food101" 101 "newExp_food101_level${3}" $2 "validation" $3

bash pretrain_script_arbitraryLevel.sh  $1 "tfds/stanford_dogs" 120 "newExp_stanford_dogs_level${3}" $2 "test" $3

bash pretrain_script_arbitraryLevel.sh  $1 "tfds/oxford_flowers102" 102 "newExp_oxford_flowers102_level${3}" $2 "validation" $3

#bash pretrain_script_arbitraryLevel.sh  $1 "tfds/caltech_birds2011" 200 "newExp_caltech_birds2011_level${3}" $2 "test" $3

bash pretrain_script_arbitraryLevel.sh  $1 "aircraft" 100 "newExp_aircraft_level${3}" $2 "validation" $3 # seems that we only have 100, but site says 102

bash pretrain_script_arbitraryLevel.sh  $1 "tfds/caltech101" 102 "newExp_caltech101_level${3}" $2 "test" $3
