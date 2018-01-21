import os
import torch
import torch.utils.data.dataloader as dataloader 
from PIL import Image
import numpy as np
import csv

''' Just in case there is a missing image in the dataset '''
def my_collate(batch):
    batch_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    missing_batchlen = len(batch)
    while len(batch) < batch_size:
        idx = np.random.randint(missing_batchlen)
        batch.append(batch[idx])
    return dataloader.default_collate(batch)

''' Resize image to 64 x 64. This can be done on the fly though '''
def resize_images(image_dir, new_width, new_height):
    files = os.listdir(image_dir)
    for f in files :
        print (f)
        if os.path.isdir(image_dir + f):
            # go through each file f and resize images
            images = os.listdir(image_dir + f)
            for image in images:
                try : 
                    orig_img = Image.open(image_dir + f + '/' + image)
                except : 
                    continue

                resized_img = orig_img.resize((new_width, new_height), Image.ANTIALIAS)
                reesized_img.save(image_dir + f +'/' + image)

'''Too many missing images, so just delete them from csvfile '''
def remove_missing_images (image_dir, csvfilename):
    original = open(image_dir + csvfilename, 'r')
    new = open(image_dir + 'edited_' + csvfilename, 'w')
    reader = csv.reader(original)
    writer = csv.writer(new)
    for line in reader:
        path = line[0]
        try :
            img = Image.open(image_dir + path)
            writer.writerow(line)
        except Exception as e:
            print (e)

            

if __name__=='__main__':
    image_dir = '/media/bach4/kylee/anime-faces/'
    # resize_images(image_dir, 64,64)
    remove_missing_images (image_dir, 'annotations.csv')

