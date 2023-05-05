import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

from tsnecuda import TSNE
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

from tqdm import tqdm

import sys
sys.path.append(os.path.join(sys.path[0], '../'))
import aircraft

sys.path.append(os.path.join(sys.path[0], 'simsiam/simsiam'))
import loader
import builder

sys.path.append(os.path.join(sys.path[0], 'simsiam'))
from main_simsiam import ProgressMeter, AverageMeter

from dual import DualModel

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
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
parser.add_argument('--best-model', default='', type=str, metavar='MODEL',
                    help='path to best model (default: "")')
parser.add_argument('--filename', default='moo.pdf', type=str,
                    help='what do you want to name the output file (default: moo.pdf)')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--model', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--dual', action='store_true',
                    help='whether we use the dual model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
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
            args.dataset, root=args.dataset, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=the_transform)
    else:
        train_dataset = aircraft.Aircraft('./aircraft', train=False, download=args.dataset_download,
            transform=the_transform)

    print('dataset loaded')
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes)

    print('loading network')

    if args.dual:
        model = DualModel(model, args)
        model.load_state_dict(torch.load(args.best_model)['state_dict'])
        model = model.model
    else:
        model.load_state_dict(torch.load(args.best_model)['state_dict'])

    model = model.cuda()
    model.encoder = nn.Sequential(*list(model.children())[:-1])

    print('plotting tsne labels')
    plot_tsne(val_loader, model, args)

def plot_tsne(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time],
        prefix="")

    # switch to eval mode
    model.eval()

    end = time.time()

    the_embeddings = list()
    the_classes = list()
    print('about to load images')
    for i, (images, labels) in enumerate(tqdm(val_loader)):
        images, labels = images.cuda(), labels.cuda()
        # measure data loading time
        # data_time.update(time.time() - end)

        # compute output and loss
        with torch.no_grad():
            output = model(images)

        for item_idx in range(labels.shape[0]):
            curr_label = labels[item_idx].item()
            the_embeddings.append(output[item_idx].to(torch.float64).cpu().numpy())
            the_classes.append(curr_label)
        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    #print(torch.mean(torch.from_numpy(prototype_by_class), dim=0))

    # # Compute t-SNE embeddings
    the_embeddings = np.array(the_embeddings)
    tsne_embeddings = TSNE(n_components=2).fit_transform(the_embeddings)

    # # Sort t-SNE embeddings by class
    embeddings_by_class = [list() for i in range(args.num_classes)]
    # print(len(tsne_embeddings))
    # print(len(the_classes))
    assert len(tsne_embeddings) == len(the_classes)
    for i in range(len(tsne_embeddings)):
        the_class = the_classes[i]
        embeddings_by_class[the_class].append(tsne_embeddings[i])
    
    # print(len(embeddings_by_class))
    # print(len(the_classes))
    # # Plot the t-SNE embeddings
    assert len(embeddings_by_class) == args.num_classes
    for i in range(len(embeddings_by_class)):
        curr_class_embeddings = embeddings_by_class[i]
        curr_class_embeddings = np.stack(curr_class_embeddings)
        curr_class = the_classes[i]
        plt.scatter(curr_class_embeddings[:,0], curr_class_embeddings[:,1], c=np.random.rand(3,), label=i)
    #plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
    plt.savefig(args.filename)
    plt.show()

    return

if __name__ == "__main__":
    main()