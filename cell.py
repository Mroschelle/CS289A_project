#dependencies
#Anaconda 3.6
#Pytorch
#Tensorboardx 
#OpenCV python
import torch
from math import ceil, sqrt
import pickle
import torch.nn as nn
import os
from torch.autograd import Variable
# from skimage import io, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time
from Neural_Network_Class import conv_deconv #Class where the network is defined
import pdb, glob
import scipy.io as spio

load = 1
train = 0
test = 1
tests_num = 1000
writer = SummaryWriter('runs_hn')
# once done, type in "tensorboard --logdir=runs_hn --bind_all" in terminal and go to the link being shown to visualize data

class ImageDataset(Dataset): #Defining the class to load datasets

    def __init__(self,input_dir,train=True):
        self.input_dir=input_dir
        self.train=train
        self.pix_size = 71
        self.current_datain=np.array([],dtype=np.float).reshape(0,self.pix_size,self.pix_size)
        self.current_dataout=np.array([],dtype=np.float).reshape(0,self.pix_size,self.pix_size)
        self.test_packts = 1
        self.train_packts = len(glob.glob(self.input_dir+'/*.mat')) - self.test_packts
    def __len__ (self):
        if self.train:
            return self.train_packts*2000 #I have kept size of testing data to be 50
        else:
            return self.test_packts*2000

    def __getitem__(self,idx):
        if self.train:
            if idx % 2000 == 0:
                ind = (int(idx/2000))%(self.train_packts)+1
                self.current_datain = spio.loadmat(self.input_dir+'/'+str(ind)+'.mat', squeeze_me=True)['tumorImage_withPSF']
                self.current_dataout = spio.loadmat(self.input_dir+'/'+str(ind)+'.mat', squeeze_me=True)['tumorImage_noPSF']
        else:
             if idx % 2000 == 0:
                ind = (int(idx/2000))%(self.test_packts)+1
                self.current_datain =  spio.loadmat(self.input_dir+'/'+str(self.train_packts + ind)+'.mat', squeeze_me=True)['tumorImage_withPSF']
                self.current_dataout = spio.loadmat(self.input_dir+'/'+str(self.train_packts + ind)+'.mat', squeeze_me=True)['tumorImage_noPSF']   
        input_image= self.current_datain[idx%2000].reshape((1,self.pix_size,self.pix_size))     
        input_image = (input_image - input_image.min())/(input_image.max()-input_image.min())
        output_image=self.current_dataout[idx%2000].reshape((1,self.pix_size,self.pix_size))       
        output_image = (output_image - output_image.min())/(output_image.max() - output_image.min())              

        sample = {'input_image': input_image, 'output_image': output_image}             

        return sample


train_dataset=ImageDataset(input_dir="images") #Training Dataset
test_dataset=ImageDataset(input_dir="images",train=False) #Testing Dataset
batch_size = 250 #mini-batch size
n_iters = 24000 #total iterations
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = ceil(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
model=conv_deconv().cuda(1) # Neural network model object

iter=0
iter_new=0 
check=os.listdir("checkpoints") #checking if checkpoints exist to resume training
if load and len(check):
    check.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    model=torch.load("checkpoints/"+check[-1],map_location=torch.device('cpu')).to(device)
    iter=int(re.findall(r'\d+',check[-1])[0])
    iter_new=iter
    print("Resuming from iteration " + str(iter))
    #os.system('python visualise.py')

                                                                              # https://discuss.pytorch.org/t/can-t-import-torch-optim-lr-scheduler/5138/6 
beg=time.time() #time at the beginning of training
if train:
    print("Training Started!")
    criterion=nn.MSELoss().cuda(1)  #Loss Class
        
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) #optimizer class
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)# this will decrease the learning rate by factor of 0.1
    for epoch in range(num_epochs):
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
        for i,datapoint in enumerate(train_loader):
            datapoint['input_image']=datapoint['input_image'].type(torch.FloatTensor) #typecasting to FloatTensor as it is compatible with CUDA
            datapoint['output_image']=datapoint['output_image'].type(torch.FloatTensor)
            input_image = Variable(datapoint['input_image'].cuda(1)) #Converting a Torch Tensor to Autograd Variable
            output_image = Variable(datapoint['output_image'].cuda(1))
            
            optimizer.zero_grad()  #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
            outputs = model(input_image)
            # loss = criterion(outputs.to(torch.device('cpu')), output_image)
            loss = criterion(outputs, output_image.cuda(1))
            loss.backward() #Backprop
            optimizer.step()    #Weight update
            writer.add_scalar('Training Loss',loss.data.item(), iter)
            iter=iter+1
            if iter % 25 == 0 or iter==1:
                # Calculate Accuracy         
                test_loss = 0
                total = 0
                # Iterate through test dataset
                for j,datapoint_1 in enumerate(test_loader): #for testing
                    datapoint_1['input_image']=datapoint_1['input_image'].type(torch.FloatTensor)
                    datapoint_1['output_image']=datapoint_1['output_image'].type(torch.FloatTensor)
                
                    input_image_1 = Variable(datapoint_1['input_image'].cuda(1))
                    output_image_1 = Variable(datapoint_1['output_image'].cuda(1))
                    
                    # Forward pass only to get logits/output
                    outputs = model(input_image_1)
                    test_loss += criterion(outputs, output_image_1).data.item()
                    total+=1 # datapoint_1['output_image'].size(0)
                test_loss= test_loss/total   #sum of test loss for all test cases/total cases
                writer.add_scalar('Test Loss',test_loss, iter) 
                # Print Loss
                time_since_beg=(time.time()-beg)/60
                print('Iteration: {}. Loss: {}. Test Loss: {}. Time(mins) {}'.format(iter, loss.data.item(), test_loss,time_since_beg))
            if iter % 500 ==0:
                torch.save(model,'checkpoints/model_iter_'+str(iter)+'.pt')
                print("model saved at iteration : "+str(iter))
                # writer.export_scalars_to_json("runs_hn/scalars.json") #saving loss vs iteration data to be used by visualise.py
        scheduler.step()        
    writer.close()          

if test:
    iter = 0
    print('Testing the %d first samples'%tests_num)
    fig = plt.figure(figsize=(30,30))
    #input_imgs = zeros(71,71,test_num)
    #output_imgs = zeros(71,71,test_num)
    #true_outs = zeros(71,71,test_num)
    #input_psf = zeros(71,71,test_num)
    #output_psf = zeros(71,71,test_num)
    for i in range(tests_num):
        # Calculate Accuracy  
        datapoint_1 = test_dataset[i]
        datapoint_1['input_image']=torch.tensor(datapoint_1['input_image']).type(torch.FloatTensor)
        datapoint_1['output_image']=torch.tensor(datapoint_1['output_image']).type(torch.FloatTensor)
   
        if torch.cuda.is_available():
            input_image_1 = Variable(datapoint_1['input_image'].to(device))
            output_image_1 = Variable(datapoint_1['output_image'].to(torch.device('cpu')))
        else:
            input_image_1 = Variable(datapoint_1['input_image'])
            output_image_1 = Variable(datapoint_1['output_image'])
        
        outputs = model(input_image_1.reshape((1,1,71,71)))
        point_src = outputs*0
        point_src[0,0,36,36] = 1
        PSF = model(point_src).reshape((71,71))
        point_src = ((point_src.cpu()).reshape((71,71))).data.numpy()
        PSF = (PSF.cpu()).data.numpy()
        time_since_beg=(time.time()-beg)/60
        print('Iteration: {}. Time(mins) {}'.format(iter, time_since_beg))           
        # plt.subplot(221)
        # plt.imshow(((input_image_1.cpu()).reshape((71,71))).data.numpy())
        # plt.subplot(222)
        # plt.imshow(((outputs.cpu()).reshape((71,71))).data.numpy())
        # plt.subplot(224)
        # plt.imshow(((output_image_1.cpu()).reshape((71,71))).data.numpy())
        # plt.savefig('test_results/' + str(iter) + '.png') 
        iter = iter + 1  
        spio.savemat('test_results/data_'+ str(iter) + '.mat', {     'input_image':((input_image_1.cpu()).reshape((71,71))).data.numpy(),
                                                        'output':((outputs.cpu()).reshape((71,71))).data.numpy(),
                                                        'gnd_truth':((output_image_1.cpu()).reshape((71,71))).data.numpy()})
    plt.close('all')
    fig = plt.figure(figsize=(30,30))
    plt.subplot(121)
    plt.imshow(point_src)
    plt.subplot(122)
    plt.imshow(PSF)
    plt.savefig('test_results/PSF' + '.png')
    plt.close('all')

writer.close()
#decrease learning rate
















