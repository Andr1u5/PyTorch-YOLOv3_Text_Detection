import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    pad = torch.from_numpy(np.array(pad))
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, image_path, label_path, img_size=416, augment=True, multiscale=False, normalized_labels=False): #Augmentation removed and normalized labels removed, img_size changed from 416 to 200
        

        img_list = []
        lbl_list = []
       
        for i in range(len(os.listdir(image_path))):
            img_f_name = "img_{}.jpg".format(i+1)
            lbl_f_name = "gt_img_{}.txt".format(i+1)
            img_list.append(img_f_name)
            lbl_list.append(lbl_f_name)
        self.img_files = img_list
        self.label_files = lbl_list
        self.img_path = image_path
        self.label_path = label_path
            

        #
        #with open(list_path, "r") as file:
        #   self.img_files = file.readlines()

        #self.label_files = [
        #    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
         #   for path in self.img_files
        #]
        
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 5 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0


    def __getitem__(self, index):
        
        # ---------
        #  Image
        # ---------
       
        img_path = self.img_files[index % len(self.img_files)]#.rstrip()
       
        # link path and file names
        img_path = self.img_path+img_path
        """ ######################  """
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        #print("----------image testing for index {}---------------".format(index))
        #print("label path: ", img_path)

        _, h, w = img.shape
        #print("original image shape ", img.shape)
        #print(h,w)
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        #print("padded img shape ", img.shape)
        #print(padded_h, padded_w)

        # ---------
        #  Label
        # ---------

        lbl_path = self.label_files[index % len(self.img_files)]#.rstrip()
        #print("labels index ", index % len(self.img_files))
        lbl_path = self.label_path+lbl_path
        targets = None
        if os.path.exists(lbl_path):
            #print(lbl_path)
            #print(img_path)
            imported_txt = torch.from_numpy(np.loadtxt(lbl_path, delimiter=",", usecols = (0,1,2,3,4,5,6,7), encoding = "utf-8-sig", dtype = 'float64').reshape(-1, 8))
            #print("------------------Image Label Testing for index {}--------------------------------".format(index))
            #print("label path: ", lbl_path)
            #print("imported text")
            #print(imported_txt)
            boxes = torch.zeros((len(imported_txt), 5))
            for i in range(len(imported_txt)):
                x_max = max(imported_txt[i][0],imported_txt[i][2],imported_txt[i][4],imported_txt[i][6])
                x_min = min(imported_txt[i][0],imported_txt[i][2],imported_txt[i][4],imported_txt[i][6])
                y_max = max(imported_txt[i][1],imported_txt[i][3],imported_txt[i][5],imported_txt[i][7])
                y_min = min(imported_txt[i][1],imported_txt[i][3],imported_txt[i][5],imported_txt[i][7])
                height = y_max - y_min
                width = x_max - x_min
                boxes[i][1] = x_min # for top left x axis
                boxes[i][2] = y_min # for top left y axis
                boxes[i][3] = width
                boxes[i][4] = height

            #print("boxes")
            #print(boxes)

            center_x = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            center_y = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            #print("unpadded center x")
            #print(center_x)
            #print("unpadded center y")
            #print(center_y)
            #print("padding")
            #print(pad)
            #width = boxes[:,3]
            #height = boxes[:,4]
            #print(type(center_x))
            #print(type(pad[2]))
            center_x=torch.add(center_x,float(pad[0]))
            center_y=torch.add(center_y,float(pad[2]))
            # center_x = center_x + pad[0]
            # center_y = center_y + pad[2]
            
            boxes[:, 1] = center_x / padded_w
            boxes[:, 2] = center_y / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            #print(boxes)
            #for i in range(0,boxes.shape[0]):
            #    if(boxes[i,1]>1):
            #        print("w")
            #        print(w_factor)
            #        print(boxes[i,1])
            #        print( padded_w)
            #        print(center_x[i], boxes[i,3])
            #        exit()
            #for j in range(0,boxes.shape[0]):
            #    if(boxes[j,2]>1):
            #        print("h")
            #        print(boxes[j,2])
            #        print(padded_h)
            #        print(center_y[i], boxes[i,4])
            #        exit()
            #print("-----------------img label testing end------------------------")
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
            
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
