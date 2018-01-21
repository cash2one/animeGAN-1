import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from model import Generator, Discriminator, AnimeFaceDataset 
from utils import my_collate

writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print(args)
args.cuda = torch.cuda.is_available()

batch_size = 128
learning_rate = 0.0002
num_epochs = 50
feature_dim = 100
smooth_label = 0.1 # smoothing the discriminator labels (1->0.9, 0->0.1)

image_dir = '/media/bach4/kylee/anime-faces/'
base_dir = '/home/kylee/dev/animeGAN-pytorch/'

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(Generator, Discriminator, dataloader, criterion):
    # setup optimizer 
    g_optim = torch.optim.Adam(Generator.parameters(), lr=learning_rate, betas=(0.5,0.999))
    d_optim = torch.optim.Adam(Discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999))
    
    # initialize noise 
    noise = torch.randn(batch_size, feature_dim, 1, 1)
    test_noise = torch.randn(batch_size, feature_dim, 1, 1)
    true_label = torch.FloatTensor(batch_size).fill_(1 - smooth_label)
    true_label_g = torch.FloatTensor(batch_size).fill_(1)
    fake_label = torch.FloatTensor(batch_size).fill_(0 + smooth_label)

    if args.cuda:
        noise = noise.cuda()
        test_noise = test_noise.cuda()
        true_label = true_label.cuda()
        true_label_g = true_label_g.cuda()
        fake_label = fake_label.cuda()
    
    # train!
    Generator.train()
    Discriminator.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            img = data['img']
            # label = data['label']

            if args.cuda:
                img = img.cuda()
                # label = label.cuda()
            
            img_real = Variable(img)
            # label = Variable(label)
            
            #------------------------#
            # Train  Discriminator
            #------------------------#
            Discriminator.zero_grad()
            # learn real data as real
            true_label_var = Variable(true_label)
            out_real = Discriminator(img_real)
            loss_d_real = criterion(out_real, true_label_var)
            
            # learn fake data as fake 
            noise_var = Variable(noise)
            # generate fake data with generator
            img_fake = Generator(noise_var)
            fake_label_var = Variable(fake_label)
            out_fake = Discriminator(img_fake.detach())
            loss_d_fake = criterion(out_fake, fake_label_var)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            d_optim.step()
            
            #-------------------#
            # Train Generator 
            #-------------------#
            Generator.zero_grad()
            # fool discriminator to learn as real with fake data
            true_label_g_var = Variable(true_label_g)
            out_real_but_fake = Discriminator(img_fake)
            loss_g = criterion(out_real_but_fake, true_label_g_var)
            loss_g.backward()
            g_optim.step()

            print ("Epoch [%d/%d] Iter [%d/%d] Loss D : %.4f, Loss G : %.4f, D(x) : %.4f, D(z) : %.4f, g : %.4f"%(epoch+1, num_epochs, i+1, len(dataloader), loss_d.data[0], loss_g.data[0], loss_d_real.data.mean(), loss_d_fake.data.mean(), loss_g.data.mean()))

            niter = epoch * len(dataloader) + i+1
            writer.add_scalar('Loss/D', loss_d.data[0], niter)
            writer.add_scalar('Loss/G', loss_g.data[0], niter)
            writer.add_scalar('D/D(x)', loss_d_real.data.mean(), niter)
            writer.add_scalar('D/D(z)', loss_d_fake.data.mean(), niter)
            writer.add_scalar('D/g', loss_g.data.mean(), niter)
            
            # generate on the way
            if (i+1)%100 == 0 :
                test_noise_var = Variable(test_noise)
                test_img = Generator(test_noise_var)
                vutils.save_image(test_img.data, base_dir + 'fake_img_epoch_%d_iter_%d.png'%(epoch+1, i+1), normalize=True)
                writer.add_image('fake_images', vutils.make_grid(test_img.data, normalize=True), niter)


        # save model
        torch.save(Generator.state_dict(), 'Generator.pth')
        torch.save(Discriminator.state_dict(), 'Discriminator.pth')
    
    writer.close()

def main():
    # load data
    annotationfile = image_dir + 'edited_annotations.csv'
    animefacedata = AnimeFaceDataset(annotationfile, image_dir)
    dataloader = DataLoader(animefacedata, batch_size=batch_size, shuffle=True, collate_fn=my_collate, drop_last=True)
    print ("Data loaded : %d"%(len(animefacedata)))

    G = Generator()
    D = Discriminator()
    G.apply(init_weight)
    D.apply(init_weight)

    if args.cuda:
        G = G.cuda()
        D = D.cuda()
    
    criterion = nn.BCELoss()
    print ("Start Training")
    train(G, D, dataloader, criterion)
    print ("Finished training!")


if __name__=='__main__':
    main()
