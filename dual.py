import torch
import torch.nn as nn
import numpy as np
import pdb 
import proxynca
import losses

class DualModel(nn.Module):
    def __init__(self, model,args, bottleneck=64):

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

        # self.decoder = nn.Sequential(
        #     nn.BatchNorm1d(self.fc.in_features),
        #     nn.ReLU(),
        #     nn.Linear(self.fc.in_features, self.fc.in_features, bias=False),
        #     nn.BatchNorm1d(self.fc.in_features),
        #     nn.ReLU(),
        #     nn.Linear(self.fc.in_features, self.fc.in_features)
        # )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.fc.in_features),
            nn.Linear(self.fc.in_features, bottleneck),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(bottleneck),
            nn.Linear(bottleneck, bottleneck*2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(bottleneck*2),
            nn.Linear(bottleneck*2, bottleneck*4),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(bottleneck*4),
            nn.Linear(bottleneck*4, 4096),
            #nn.Sigmoid()
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 4096),
        )

        self.task_modules = nn.ModuleList([self.fc,self.decoder])
        self.shared_modules = model
        
        self.old = nn.ModuleList([self.fc,self.model]) # pretrained, original task, original weights
        self.new = nn.ModuleList([self.decoder]) # new task, new weights

        return
    
    def forward(self,x,on=False):
        x = self.model(x)
        x1 =  self.fc(x)
        if on:
            x2 = self.decoder(x)
            return x1, x2
        return x1

class DualLoss(nn.Module):
    # def construct_dense_labels(self, num_classes):
    #     the_array = np.zeros((num_classes, 64*32))
    #     for i in range(64*32):
    #         the_array[min(i // 20, 99), i] = 1
    #     return the_array.astype("float32")
    def __init__(self,loss,weights,num_classes):
        super(DualLoss, self).__init__()
        # TODO: change dense loss
        self.dense_loss = nn.BCEWithLogitsLoss()
        self.categorical_loss = loss
        # TODO: change dense labels
        # labels are contrastively learned, but they are floats
        # dense_embeddings = np.load('simsiam/cifar_prototypes.npy')
        # medians = np.median(dense_embeddings, axis=0)
        # dense_binary_embeddings = np.where(dense_embeddings > medians, 1, 0)
        dense_binary_embeddings = np.random.choice([0, 1], size=(num_classes,64*64)).astype("float32")
        import matplotlib.pyplot as plt
        plt.imsave('dense binary embeddings.png', dense_binary_embeddings, cmap='gray')
        #plt.show()
        self.dense_labels = torch.tensor(dense_binary_embeddings.astype('float32'))
        self.num_classes = num_classes
        self.weights = weights
    
    def forward(self,output,target,seperate=False):
        dense_target = []
        for t in target:
            dense_target.append(self.dense_labels[t])
        dense_target = torch.stack(dense_target).cuda()
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1], dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2

    def update_labels(self):
        return