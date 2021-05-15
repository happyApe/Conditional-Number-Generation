import os
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image


os.makedirs('cgan_images',exist_ok = True)


num_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
z_dims = 100
num_classes = 10
img_size = 32
channels = 1
sample_interval = 400 # interval between image sampling


img_shape = (channels,img_size,img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.label_emb = nn.Embedding(num_classes,num_classes)

        self.main = nn.Sequential(

                nn.Linear(z_dims + num_classes, 128),
                nn.LeakyReLU(0.2,inplace = True),

                nn.Linear(128,256),
                nn.BatchNorm1d(256,0.8),
                nn.LeakyReLU(0.2,inplace = True),

                nn.Linear(256,512),
                nn.BatchNorm1d(512,0.8),
                nn.LeakyReLU(0.2,inplace = True),

                nn.Linear(512,1024),
                nn.BatchNorm1d(1024,0.8),
                nn.LeakyReLU(0.2,inplace = True),

                nn.Linear(1024,int(np.prod(img_shape))), # 1 x 32 x 32
                nn.Tanh()
        )

    def forward(self,noise,labels):
        # concatenate the label embedding and the input
        gen_input = torch.cat((self.label_emb(labels),noise),-1)
        img = self.main(gen_input)
        img = img.view(img.size(0),*img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.label_emb = nn.Embedding(num_classes,num_classes)

        self.main = nn.Sequential(

                nn.Linear(num_classes + int(np.prod(img_shape)),512),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Linear(512,512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Linear(512,512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Linear(512,1)
        )


    def forward(self,img,labels):
        disc_input = torch.cat((img.view(img.size(0),-1),self.label_emb(labels)),-1)
        validity = self.main(disc_input)
        return validity


# Loss functions
critertion = torch.nn.MSELoss()


# Initialize the adversaries

gen = Generator()
disc = Discriminator()

if cuda:
    gen.cuda()
    disc.cuda()
    critertion.cuda()

# Data

dataloader = DataLoader(datasets.MNIST('data',train = True,download = True,
                    transform = transforms.Compose(
                        [transforms.Resize(img_size),transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])])),
                        batch_size = batch_size, shuffle = True)

# Optimizers

optim_gen = optim.Adam(gen.parameters(),lr = lr,betas = (b1,b2))
optim_disc = optim.Adam(disc.parameters(),lr = lr,betas = (b1,b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
    "Saving a grid of generated digits ranging form 0 to num_classes"
    # sample noise
    z = Variable(FloatTensor(np.random.normal(0,1,(n_row**2,z_dims))))

    # Getting labels from 0 to num_classes for n_rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = gen(z,labels)
    save_image(gen_imgs.data,"cgan_images/%d.png" % batches_done, nrow = n_row,normalize = True)



## Training ##

for epoch in range(num_epochs):

    for i,(imgs,labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)


        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))


        ## Train generator

        optim_gen.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0,1,(batch_size,z_dims))))
        gen_labels = Variable(LongTensor(np.random.randint(0,num_classes,batch_size)))

        gen_imgs = gen(z,gen_labels)

        validity = disc(gen_imgs,gen_labels)
        g_loss = critertion(validity,valid)

        g_loss.backward()
        optim_gen.step()


        ## Train discriminator

        optim_disc.zero_grad()

        validity_real = disc(real_imgs,labels)
        d_real_loss = critertion(validity_real,valid)

        validity_fake = disc(gen_imgs.detach(),gen_labels)
        d_fake_loss = critertion(validity_fake,fake)
        
        d_loss = (d_real_loss + d_fake_loss)/2

        d_loss.backward()
        optim_disc.step()

        if i % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        batches_done = epoch*len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row = 10,batches_done = batches_done)
    
