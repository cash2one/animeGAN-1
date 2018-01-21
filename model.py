import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(100, 1024, 4, stride=1, padding=0),
                nn.BatchNorm2d(1024),
                #nn.LeakyReLU(0.02, inplace=True),
                nn.ReLU(),
                nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                #nn.LeakyReLU(0.02, inplace=True),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                #nn.LeakyReLU(0.02, inplace=True),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                #nn.LeakyReLU(0.02, inplace=True),
                nn.ReLU(),
                nn.ConvTranspose2d(128,3, 4, stride=2, padding=1),
                nn.Tanh())


    def forward(self, x):
        out = self.deconv(x)
        return out 


class Discriminator(nn.Module): 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(3, 128, 4, stride=2, padding=1),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1), 
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Conv2d(256, 512, 4,stride=2, padding=1), 
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Conv2d(512, 1024, 4, stride=2, padding=1), 
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Conv2d(1024, 1, 4, stride=1, padding=0),
                nn.Sigmoid())


    def forward(self,x):
        # image size is 3 x 96 x 96 
        out = self.conv(x)
        out = out.view(-1,1).squeeze(1)
        return out 
    
class AnimeFaceDataset(Dataset):
    def __init__(self, csvfile, data_dir):
        anime_file = open(csvfile, 'r')
        self.lines = anime_file.readlines()[1:]
        self.data_dir = data_dir 
    
    def __getitem__(self, index):
        line = self.lines[index].strip().split(',')
        img_path = line[0]
        img_label = line[1]
        
        try : 
            img = Image.open(self.data_dir + img_path)
        except Exception as e :
            return
        
        # transform image : to tensor and normalize
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        img = transform(img) # type Tensor


        entry = {'img' : img, 'label': img_label}
        return entry


    def __len__(self):
        return len(self.lines)
        


