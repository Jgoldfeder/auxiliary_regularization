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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    model.load_state_dict(torch.load('best_contrastive_label_network.pth'))

    model = model.encoder

    generate_labels(train_loader, model, args)

def generate_labels(train_loader, model, args):
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
    np.save('cifar_prototypes.npy', prototype_by_class)

    # Compute t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2).fit_transform(prototype_by_class)
    # Plot the t-SNE embeddings
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
    plt.savefig('tsne_embeddings.png')
    #plt.show()

    return

if __name__ == "__main__":
    main()