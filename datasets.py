import pickle
# import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import datasets as tvdatasets
from torchvision import transforms
import json

# IMAGENET100_PATH = "D:\Code\Datasets\ImageNet100\imagenet-100"

def load_imagenet100_data(path):
    train_folder = path + "/train"
    test_folder = path + "/val"
    label_json_file = path + "/Labels_100.json"
    with open(label_json_file, "r") as f:
        label_names = json.load(f)
    train_dataset = tvdatasets.ImageFolder(train_folder,
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                           ]))
    test_dataset = tvdatasets.ImageFolder(test_folder,
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                            ]))
    return train_dataset, test_dataset, label_names



if __name__=='__main__':
    pass
    # train_dataset, test_dataset, label_names = load_imagenet100_data(IMAGENET100_PATH)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # print(len(train_dataset),len(test_dataset))
    # print(len(train_loader),len(test_loader))
    # for i,(img,label) in enumerate(train_loader):
    #     print(img.shape,label.shape)
    #     print(label[0])
    #     # cv2.imshow('test',img[0].numpy().transpose(1,2,0))
    #     # #cv2.imshow('test',cv2.resize(cv2.cvtColor(img[0].numpy(), cv2.COLOR_RGB2BGR),(224,224)))
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     break