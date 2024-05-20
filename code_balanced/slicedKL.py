import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# class Net_S(nn.Module):
#     def __init__(self,dim,H):
#         super(Net_S, self).__init__()

#         self.fc1 = nn.Linear(dim, H)
#         #self.fc2 = nn.Linear(dim, H)
#         self.fc3 = nn.Linear(H, 1)

#     def forward(self, x):
#         h1 = F.relu(self.fc1(x))
#         h2 = self.fc3(h1)
#         return h2 
    

class slicedKLclass(nn.Module):
    def __init__(self,d,n_slice,k,device):
        super(slicedKLclass,self).__init__()
        #Parallel
        self.k = k            #slice dim
        self.n_slice = n_slice
        self.d = d            
        H = 30*self.k
        
        self.fc1 = nn.Conv1d(in_channels=self.n_slice * self.k, out_channels=self.n_slice * H, kernel_size=1, groups=self.n_slice)
        self.fc3 = nn.Conv1d(in_channels=self.n_slice * H, out_channels=self.n_slice, kernel_size=1, groups=self.n_slice)

        
        # self.fc1 = []
        # self.fc3 = []
        # for k in range(n_slice):
        #     # nt = Net_S(K,H)
        #     # if torch.cuda.is_available():
        #     #     nt.cuda()
        #     # self.nets.append(nt)
        #     self.fc1.append(nn.Linear(self.k,H))
        #     self.fc3.append(nn.Linear(H,1))
                            
        #Generate slices
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Generate your slices once for all 
        self.thetas = torch.from_numpy(self.gen_k_slice()).type(torch.FloatTensor).to(device)
        #self.thetas = torch.from_numpy(self.gen_k_slice()).type(torch.FloatTensor)
        #phis = torch.from_numpy(gen_k_slice(dim,n_slice,k)).type(torch.FloatTensor).to(device)
#         self.optimizers = []
#         for i in range(n_slice):

#             self.optimizers.append(torch.optim.Adam(list(self.net[i].parameters()), lr=lr))
    
    def gen_slice(self,d=1, n=1 ):
        xx = np.random.normal(0.,1.,[n,d])
        return xx/(np.expand_dims(np.linalg.norm(xx,axis=1),axis = 1) @ np.ones((1,d)))
    def gen_k_slice(self):
        xx = np.random.normal(0.,1.,[self.n_slice,self.k,self.d])
        slices = np.zeros((self.n_slice,self.k,self.d))
        for i in range(self.n_slice):
            uu,ss,slices[i,:,:] = np.linalg.svd(xx[i,:,:], full_matrices = False)
        return slices

    def kl_criterion(self,pred_x, pred_y):
        # Donsker-varadhan
        ret = torch.mean(pred_x,axis=0) - torch.logsumexp(pred_y - np.log(pred_y.shape[0]), axis=0) #torch.log(torch.mean(torch.exp(pred_y),axis=0))
        loss = - ret  # maximize
        return loss
    def forward(self,X,Y,X_noise, sigma_noise=0):
        loss = 0
        
        x_sample = torch.zeros((X.shape[0], self.k * self.n_slice,1)).cuda()
        y_sample = torch.zeros((Y.shape[0], self.k * self.n_slice,1)).cuda()
        ## PROBABLY SLOW, MAYBE SPEED THIS UP
        for k in range(self.n_slice):
           
            x_sample[:,k*self.k:(k+1)*self.k,0] = X @ torch.transpose(self.thetas[k,:,:], 0,1)
            y_sample[:,k*self.k:(k+1)*self.k,0] = Y @ torch.transpose(self.thetas[k,:,:], 0,1) 
            #y_sample = Y @ torch.transpose(self.thetas[k,:,:],0,1) 
            
        # Add noise for privacy
        if sigma_noise > 0:
            if X_noise is not None:
                
                
                x_sample[:,:,0] = x_sample[:,:,0] + sigma_noise * X_noise #torch.randn(x_sample.shape).cuda()
            else:
                x_sample = x_sample + sigma_noise * torch.randn(x_sample.shape).cuda()
            y_sample = y_sample + sigma_noise * torch.randn(y_sample.shape).cuda()
        
            #More MINE things
        h1 = F.relu(self.fc1(x_sample))
        pred_x = self.fc3(h1)
        h1y = F.relu(self.fc1(y_sample))
        pred_y = self.fc3(h1y)
            # pred_x = net[slice_ix](x_sample)
            # pred_y = net[slice_ix](y_sample)

        loss = torch.mean(self.kl_criterion(pred_x, pred_y)) #/self.n_slice
        return loss
#     def train(self, trainloader,epoch, n_slice, net, optimizers,thetas):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         #Number of slices: n_slice = m in the paper
#         #Parallel neural networks, one for each slice - quite likely a faster way to set this up
#         for i in range(n_slice):
#             net[i].train()
#         train_loss = np.zeros(n_slice)
#         counts = np.zeros(n_slice)
#         for batch_idx, (X, Y) in enumerate(trainloader):
#             #For each minibatch, update a random "num_update" of the slices
#             slices = np.random.permutation(n_slice)
#             num_update = 100
#             for k in range(num_update): #For each slice
#                 slice_ix = slices[k]
#                 counts[slice_ix] += 1
#                 batchsz = len(X)

#                 #Do the projection on the slice
#                 x_sample = X @ torch.transpose(thetas[slice_ix,:,:], 0,1) 
#                 y_sample = Y @ torch.transpose(thetas[slice_ix,:,:],0,1) 



            
            
#             #More MINE things
#             pred_x = net[slice_ix](x_sample)
#             pred_y = net[slice_ix](y_sample)

#             loss = kl_criterion(pred_x, pred_y)

#             train_loss[slice_ix] += loss
#             #Update the slice
#             net[slice_ix].zero_grad()
#             loss.backward()
#             optimizers[slice_ix].step()
#     normed = train_loss/(counts)

#     return np.nanmean(normed) #np.sum(train_loss)/np.sum((counts + 0.0001))#train_loss/batch_idx


#     def slicedKL(trainloader,net,dim,k, n_epoch, n_slice, lr):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # Generate your slices once for all 
#         thetas = torch.from_numpy(gen_k_slice(dim,n_slice,k)).type(torch.FloatTensor).to(device)
#         #phis = torch.from_numpy(gen_k_slice(dim,n_slice,k)).type(torch.FloatTensor).to(device)
#         optimizers = []
#         for i in range(n_slice):

#             optimizers.append(torch.optim.Adam(list(net[i].parameters()), lr=lr))

#         train_loss = np.zeros((n_epoch))
#         for epoch in range(n_epoch):
#             if epoch % 10 == 0:
#                 print('Epoch: %d' % epoch)
#             batchsz = trainloader.batch_size
#             ix = np.random.permutation(n_slice)
#             #for slice_ix in ix[:min(n_slice,10*batchsz)]:
#             train_loss[epoch] = train(trainloader,epoch,n_slice,net,optimizers,thetas)
#             if (epoch + 1) % 100 == 0:
#                 lr = lr/5
#                 for i in range(n_slice):
#                     optimizers[i] = torch.optim.Adam(list(net[i].parameters()), lr=lr)



#         return -np.mean((train_loss[-min(20,n_epoch):]))  #this is the k-SMI