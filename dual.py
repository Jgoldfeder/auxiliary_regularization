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
            nn.ReLU(),
            nn.Linear(self.fc.in_features, self.fc.in_features, bias=False),
            nn.BatchNorm1d(self.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.fc.in_features, self.fc.in_features)
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
    def __init__(self,loss,weights,num_classes):
        super(DualLoss, self).__init__()
        # TODO: change dense loss
        self.dense_loss = nn.CrossEntropyLoss()
        self.categorical_loss = loss
        # TODO: change dense labels
        self.dense_labels = torch.tensor(np.load('simsiam/cifar_prototypes.npy'))
        self.num_classes = num_classes
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

    def update_labels(self):
        return