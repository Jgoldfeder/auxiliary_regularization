import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

from main_simsiam import ProgressMeter, AverageMeter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cifar100_coarse_to_fine = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dir', metavar='DIR', default='cifar100/train',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')

def main():
    args = parser.parse_args()

    train_dataset = datasets.CIFAR100(download=True, train=True, root=args.dir,\
                                transform=transforms.ToTensor())

    #print(train_dataset.classes)
    cifar100_label_to_idx = {label: idx for idx, label in enumerate(train_dataset.classes)}
    coarse_label_to_idx = {key: idx for idx, key in enumerate([x for x in cifar100_coarse_to_fine])}
    fine_to_coarse = [None for i in range(100)]
    for curr_coarse_label in cifar100_coarse_to_fine:
        the_curr_fine_labels = cifar100_coarse_to_fine[curr_coarse_label]
        for curr_fine_label in the_curr_fine_labels:
            curr_fine_index = cifar100_label_to_idx[curr_fine_label]
            fine_to_coarse[curr_fine_index] = coarse_label_to_idx[curr_coarse_label]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    model.load_state_dict(torch.load('best_contrastive_label_network.pth'))

    model = model.encoder

    generate_labels(train_loader, model, fine_to_coarse, args)

def generate_labels(train_loader, model, fine_to_coarse, args):
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix="")

    # switch to eval mode
    model.eval()

    end = time.time()

    embeddings_by_class = [list() for i in range(100)]
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output and loss
        output = model(images)

        for i in range(labels.shape[0]):
            curr_label = labels[i].item()
            embeddings_by_class[curr_label].append(output[i])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    # save class prototypes
    prototype_by_class = [torch.mean(torch.stack(embeddings, dim=0), dim=0) \
        for embeddings in embeddings_by_class]
    prototype_by_class = torch.stack(prototype_by_class, dim=0).detach().numpy()
    print(prototype_by_class.shape)
    #np.save('cifar_prototypes.npy', prototype_by_class)
    """
    prototype_by_class = np.load('cifar_prototypes.npy')
    colors = []
    for r in [0.1, 0.5, 1]:
        for g in [0.1, 1]:
            for b in [0.1, 0.5, 1]:
                colors.append((r, g, b))
    colors.append((0.75, 0.75, 0.5)); colors.append((0.25, 0.25, 0.5))

    # # Compute t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2).fit_transform(prototype_by_class)
    # # Plot the t-SNE embeddings
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], \
        c=[colors[fine_to_coarse[i]] for i in range(100)])
    plt.savefig('tsne_embeddings.png')
    plt.show()

    return

if __name__ == "__main__":
    main()