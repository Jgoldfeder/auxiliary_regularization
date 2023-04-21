import torch
import torch.nn as nn
import numpy as np
import pdb 
import proxynca
import losses
import copy
import random

class State:
    def __init__(self,model,opt,loss):
        self.model = model
        self.opt = opt
        self.loss = loss

        self.model_state = copy.deepcopy(model.state_dict())
        self.opt_state = []
        for o in opt:
            self.opt_state.append(copy.deepcopy(o.state_dict()))
        self.labels = copy.deepcopy(loss.dense_labels)
        self.train_loss = -1
        self.val_loss = -1

    def copy(self):
        s =  State(self.model,self.opt,self.loss)
        s.labels = copy.deepcopy(self.labels)
        s.train_loss = self.train_loss
        s.val_loss = self.val_loss
        return s
     
    def restore(self):
        self.model.load_state_dict(self.model_state)
        for i,o in  enumerate(self.opt):
            o.load_state_dict(self.opt_state[i])
        self.loss.dense_labels = self.labels
    
    def random_label(self):
        num_classes = self.labels.shape[0]
        self.labels = torch.tensor(np.random.choice([0, 1], size=(num_classes,64*64)).astype("float32"))

    def mutate(self,percent):
        for i in range(self.labels.shape[0]):
            for j in range(self.labels.shape[1]):
                if random.uniform(0, 1) > percent:
                    continue
                if self.labels[i][j] == 0:
                    self.labels[i][j] = 1
                else:
                    self.labels[i][j] = 0
        return self
    
    def save_model(self):
        self.model_state = copy.deepcopy(self.model.state_dict())

    def save_opt(self):
        self.opt_state = []
        for o in self.opt:
            self.opt_state.append(copy.deepcopy(o.state_dict()))


class DualModel(nn.Module):
    def __init__(self, model,args,bottleneck=64):

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
            nn.Linear(self.fc.in_features, 4096),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096), 
        )

        self.task_modules = nn.ModuleList([self.fc,self.decoder])
        
        self.old = nn.ModuleList([self.fc,self.model])
        self.new = nn.ModuleList([self.decoder])

        self.shared_modules = model
    

    def forward(self,x,on=False):
        x = self.model(x)
        x1 =  self.fc(x)
        if on:
            x2 = self.decoder(x)
            return x1, x2
        return x1
    


class DualModelExploratory(nn.Module):
    def __init__(self, model,args,bottleneck=64):

        super(DualModelExploratory, self).__init__()
  
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


        self.decoder2 = nn.Sequential(
            nn.BatchNorm1d(self.fc.in_features),
            nn.Linear(self.fc.in_features, 4096),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096), 
            #nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(4096),
            #nn.Linear(4096, 4096), 
        )

 
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
        self.bfc = nn.Linear(4096,4096)
        self.prefc = nn.Linear(2048, 64)
        self._smalldecoder = nn.Linear(self.fc.in_features, 4096)
        self.task_modules = nn.ModuleList([self.fc,self.decoder2])
        
        self.old = nn.ModuleList([self.fc,self.model])
        self.new = nn.ModuleList([self.decoder,self.decoder2])

        self.shared_modules = model
        
        embed_dim = 64#self.fc.in_features
        print("embed dim:",embed_dim)
        self.small = nn.Linear(self.fc.in_features, embed_dim)
        self.classify = nn.Linear(1*embed_dim, 100)
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2,
            dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, 
            kdim=embed_dim, vdim=embed_dim, batch_first=True
        ) 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)

    def forward_exploratory(self,x,on=False):
        x = self.model(x)
        x = self.small(x)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        e=x
        p = self.criterion.dense_loss.proxies

        pure_attention=True
        if pure_attention:
            x,c = self.attention(x,p,p)
            if on:
                return c,e
            return c
        else:
            outs=[]
            for x_i in x:
                combined = torch.cat([x_i.unsqueeze(0),p],0)
                out = self.encoder_layer(combined)[0]
                outs.append(out)
            x = torch.stack(outs)


        #x=torch.cat([e,x],-1)
        c = self.classify(x)
        if on:
            return c,e
        return c

    def forward(self,x,on=False):
        x = self.model(x)
        x1 =  self.fc(x)
        if on:
            x2 = self.decoder2(x)
            
            if False:
                x2 =  self.prefc(x)
                x2 = torch.unsqueeze(x2, -1)
                x2 = torch.unsqueeze(x2, -1)
                x2 = torch.reshape(self.deconvs(x2), (-1,4096))
                x2 = self.bfc(x2) 
            return x1, x2
        return x1



class DualLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes, accumulate = True):
        super(DualLoss, self).__init__()
        self.dense_loss = nn.BCEWithLogitsLoss()#nn.MSELoss()# #nn.BCELoss()
        self.categorical_loss = loss
        self.dense_labels = torch.tensor(np.random.choice([0, 1], size=(num_classes,64*64)).astype("float32"))
        self.num_classes = num_classes
        self.weights = weights
        self.accumulate = accumulate
        self.clear_sum()

    def clear_sum(self):
        self.dense_output_sum = torch.tensor(np.zeros((self.num_classes,64*64)).astype("float64"))
        self.total_accumulated = np.zeros(self.num_classes)

    def get_avg(self):
        for t in range(self.num_classes):
            if self.total_accumulated[t] != 0:
                self.dense_output_sum[t] /= self.total_accumulated[t]
        return_val = self.dense_output_sum
        self.clear_sum()
        return return_val
    
    def update_labels(self):
        avg = self.get_avg()
        #print(avg)
        current = self.dense_labels
        num_flips = 40

        for j in range(self.num_classes):
            for _ in range(num_flips):
                j_avg = avg[j,:]
                j_current = current[j,:]
                error =(j_current-j_avg).abs()
                lowest_error = torch.argmin(error)
                highest_error = torch.argmax(error)
                #j_current[lowest_error] = 1 - j_current[lowest_error]
                j_current[highest_error] = 1 - j_current[highest_error]
        #self.dense_labels = torch.tensor(np.random.choice([0, 1], size=(self.num_classes,64*64)).astype("float32"))

    def forward(self,output,target,seperate=False):
        if self.accumulate:
            for i,t in enumerate(target):
                self.dense_output_sum[t] +=  nn.Sigmoid()(output[1][i].detach()).cpu()
                self.total_accumulated[t] += 1 
        dense_target = []
        for t in target:
            dense_target.append(self.dense_labels[t])
        dense_target = torch.stack(dense_target).cuda()
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2






        
class ProxyLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes, accumulate = True):
        super(ProxyLoss, self).__init__()
        self.categorical_loss = loss
        self.num_classes = num_classes
        self.weights = weights
        
        sz_embedding = 64#2048
        self.dense_loss = proxynca.ProxyNCA(
            nb_classes = num_classes,
            sz_embedding = sz_embedding
        ).cuda()
    
    def update_labels(self):
        pass

    def forward(self,output,target,seperate=False):
        
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2


class AttModel(nn.Module):
    def __init__(self,model,class_sampler):
        super(AttModel, self).__init__()
        self.model = model
        self.class_sampler=class_sampler
        self.fc = model.fc
        embed_dim = 64
        model.fc = nn.Linear(2048,embed_dim)#nn.Identity()
        sz_embedding = embed_dim
        #self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2,
        #    dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, 
        #    kdim=embed_dim, vdim=embed_dim, batch_first=True
        #)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=sz_embedding,nhead=4, batch_first=True)
        self.conditioned_decoders = nn.ModuleList([])
        self.num_classes=len(class_sampler.sample())
        for i in range(self.num_classes):
            self.conditioned_decoders.append(nn.Linear(2*sz_embedding,1))
        self.embedding = nn.Embedding(self.num_classes,64)
        self.drop = torch.nn.Dropout(p=0.5, inplace=False) 
        self.sample =   self.class_sampler.sample()   
    def forward(self,x):
        x = self.model(x)

        num_samples = 1
        ps = []
        for _ in range(num_samples):
            #sample = self.sample#
            sample = self.class_sampler.sample()
            sample = torch.stack(sample).clone().detach()
            #pdb.set_trace()
            #with torch.no_grad():
            p = self.model(sample.float().cuda())

            ps.append(p)
        p = torch.mean(torch.stack(ps),dim=0).cuda()

       
        a = x.unsqueeze(1)
        b = p.unsqueeze(0).tile(x.shape[0],1,1)
        
        combined = torch.cat([a,b],1)
        
        x = self.encoder_layer(combined)


        query = x[:,0,:]
        conditionals = x[:,1:,:]
        query = query.unsqueeze(1).tile(1,conditionals.shape[1],1)
        #query = self.drop(query)
        combined = torch.cat([query,conditionals],2)
        slivers = []
        for i in range(self.num_classes):
            sliver = combined[:,i,:]
            sliver = self.conditioned_decoders[i](sliver).flatten()
            slivers.append(sliver)
        x = torch.stack(slivers,dim=1)


        return x
        
