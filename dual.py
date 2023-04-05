import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

class DualModel(nn.Module):
    # adopted from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    def create_data_augs(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(.1, 2.))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        return
    
    def __init__(self, model, args):

        super(DualModel, self).__init__()
  
        self.model = model

        # replace last layer, this varies by model name
        if "mixer" in args.model:
            self.fc = model.head
            model.head = nn.Identity()
        elif "vit" in args.model:
            self.fc = model.head
            model.head = nn.Identity()
        else:            
            self.fc = model.fc
            model.fc = nn.Identity()

        self.create_data_augs()

        # From https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
        dim = 2048
        pred_dim = 512
        # build a 3-layer projector
        prev_dim = self.fc.weight.shape[1]
        self.simsiam_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.simsiam_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.simsiam_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.task_modules = nn.ModuleList([self.fc, self.simsiam_projector, self.simsiam_predictor])

        self.shared_modules = model
        
        self.old = nn.ModuleList([self.fc,self.model]) # pretrained, original task, original weights
        
        self.new = nn.ModuleList([self.simsiam_predictor, self.simsiam_projector, model]) # new task, new weights
        
        return

    def forward(self,x,on=False):
        x_main = self.fc(self.model(x))
        if on:
            x1 = self.augmentation(x)
            x2 = self.augmentation(x)

            z1 = self.simsiam_projector(self.model(x1)) # NxC
            z2 = self.simsiam_projector(self.model(x2)) # NxC

            p1 = self.simsiam_predictor(z1) # NxC
            p2 = self.simsiam_predictor(z2) # NxC

            return x_main, p1, p2, z1.detach(), z2.detach()

        return x_main

class DualLoss(nn.Module):
    def __init__(self,loss,weights,num_classes):
        super(DualLoss, self).__init__()
        self.categorical_loss = loss
        self.simsiam_criterion = nn.CosineSimilarity(dim=1).cuda()
        self.num_classes = num_classes
        self.weights = weights
    
    def forward(self,output,target,seperate=False):
        x_main, p1, p2, z1, z2 = tuple(output)
        loss1 = self.categorical_loss(x_main,target)*self.weights[0]
        loss2 = -(self.simsiam_criterion(p1, z2).mean() + self.simsiam_criterion(p2, z1).mean()) * 0.5
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2

