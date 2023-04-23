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

import sys
sys.path.append(os.path.join(sys.path[0], '../'))
import aircraft

from main_simsiam import ProgressMeter, AverageMeter

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')

def main():
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    the_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    print('loading dataset')
    if args.dataset != "aircraft":
        train_dataset = create_dataset(
            args.dataset, root=args.dataset, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=the_transform)
    else:
        train_dataset = aircraft.Aircraft('./aircraft', train=True, download=args.dataset_download,
            transform=the_transform)

    print('dataset loaded')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    print('loading network')
    model.load_state_dict(torch.load('best_contrastive_label_network_{}_{}.pth'.format(\
        args.dataset.replace('/', ''), args.seed)))

    model = model.encoder
    model = model.to(args.gpu)

    print('generating labels')
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
    torch.no_grad()

    end = time.time()

    embeddings_by_class = [torch.zeros(args.dim) for i in range(args.num_classes)]
    count_by_class = [0 for i in range(args.num_classes)]
    print('about to load images')
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(args.gpu), labels.to(args.gpu)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output and loss
        output = model(images)

        for item_idx in range(labels.shape[0]):
            curr_label = labels[item_idx].item()
            embeddings_by_class[curr_label] += output[item_idx].to(torch.float64).cpu()
            count_by_class[curr_label] += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    for class_idx in range(args.num_classes):
        embeddings_by_class[class_idx] /= count_by_class[class_idx]

    # save class prototypes
    # prototype_by_class = [torch.mean(torch.stack(embeddings, dim=0), dim=0) \
    #     for embeddings in embeddings_by_class]
    prototype_by_class = embeddings_by_class
    prototype_by_class = torch.stack(prototype_by_class, dim=0).detach().numpy()
    print(prototype_by_class.shape)
    np.save('{}_prototypes_{}.npy'.format(args.dataset.replace('/', ''), args.seed), prototype_by_class)

    # # Compute t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2).fit_transform(prototype_by_class)
    # # Plot the t-SNE embeddings
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
    plt.savefig('tsne_embeddings_{}.png'.format(args.seed))
    plt.show()

    return

if __name__ == "__main__":
    main()