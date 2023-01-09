Instructions:

Here is an example command:

CUDA_VISIBLE_DEVICES=2 python train.py torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=vit_base_patch16_224_miil_in21k --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --checkpoint-hist=1 -d torch/cifar100 --dual --dual-weights .1 .9 --experiment "name of project" --name "name of run"

Lets break down the arguments:

CUDA_VISIBLE_DEVICES specifies which GPUs are visible.

-b is batch size

--img-size is image size. For transfer learning, needs to be same as imagenet usually.

--epochs number of epochs

--color-jitter=0 --amp  --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 
these are used for data augmentation and preprocessing. Need not be touched.

--lr the learning rate

--sched lr schedule. cosine is a good choice

--model what model to use. 

--pretrained use preloaded weights

--num-classes how many classes are in dataset

--opt the optimizer. SGD is good. Can try adam or adamw

--weight-decay weight decay

-d the dataset. if you set --download flag, it will download the dataset
to download a dataset:

right after train.py, write "torch/<dataset_name>"

then write -d "torch/<dataset_name>"

the above is for torchvision datasets. Can also use "tfds" for tensroflo and "wds"