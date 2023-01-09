import torch
import torch.nn as nn
import numpy as np

class DualModel(nn.Module):
    def __init__(self, model,args):

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
        
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.fc.in_features),
            nn.Linear(self.fc.in_features, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 4096),
            #nn.Sigmoid()
        )

        self.task_modules = nn.ModuleList([self.fc,self.decoder])
        self.shared_modules = model

    def forward(self,x,on=False):
        x = self.model(x)
        x1 =  self.fc(x)
        if on:
            x2 = self.decoder(x)
            return x1, x2
        return x1



class DualLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes):
        super(DualLoss, self).__init__()
        self.dense_loss =  nn.BCEWithLogitsLoss()#nn.BCELoss()
        self.categorical_loss = loss
        self.dense_labels = torch.tensor(np.random.choice([0, 1], size=(num_classes,64*64)).astype("float32"))

        self.weights = weights

    def forward(self,output,target,seperate=False):
        dense_target = []
        for t in target:
            dense_target.append(self.dense_labels[t])
        dense_target = torch.stack(dense_target).cuda()
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2