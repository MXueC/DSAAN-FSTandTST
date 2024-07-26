import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch

from preparedata_method4 import  diff_img
import cv2
import random

# def make_txt(root,file_name,label):
#     path = os.path.join(root, file_name)

#     data = os.listdir(path)

#     with open(os.path.join(root,"data.txt"),'a') as f:
#         for line in data:
#             f.write(line+','+str(label)+','+file_name + '\n')



transform = transforms.Compose([
    transforms.Resize([224,224])
    , transforms.ToTensor() ])


class DatasetTrain(Dataset):
    "TrainDataset, derived from torch.utils.data.DataSet"

    def __init__(self,path,seq_len = 5,transform=None) -> None:
        # self.txt_root = os.path.join(path,"data.txt")
        """seq_len """
        self.txt_root = path
        if not seq_len%2:
            self.seq_len = seq_len +1
        else:
            self.seq_len = seq_len
        with open(self.txt_root,'r') as f:
            self.data = f.readlines()

        
        grayimgs = []
        # eventimgs = []
        labels = []
        for line in self.data:
            line = line.rstrip()
             
            # grayimg,eventimg,label = line.split(',')
            grayimg,label = line.split(' ')

            grayimgs.append(grayimg)
            # eventimgs.append(eventimg)
            labels.append(label)

        self.grayimgs = grayimgs
        self.labels = labels 
        # self.eventimgs = eventimgs
        self.transform = transform

    def __len__(self):
        return len(self.labels)


    def __getitem__(self,index):
        # print("----------------INDEX :{}-----------------------".format(index))
        start_index = index-self.seq_len//2
        stop_index = index+self.seq_len//2
        # print("start_index:{},stop_index:{}".format(start_index,stop_index))
        if self.seq_len>6750:
            print("~~~~~~~~~")
        if index%6750>= (6750-self.seq_len//2):
            stop_index = 6750-index%6750-1+index
            start_index= stop_index-self.seq_len+1
        if index%6750< self.seq_len//2:
            start_index = index - index%6750
            stop_index = start_index + self.seq_len-1

        # if (index % 6750 < (6750 - self.seq_len // 2)) and (index % 6750 >= self.seq_len // 2):
        dif_img = diff_img(self.grayimgs, start_index, stop_index)
        # print(index,start_index,stop_index)
        # imgs = [np.array(cv2.imread( self.grayimgs[index],cv2.COLOR_BGR2GRAY)) for index in range(start_index,stop_index+1)]


        imgs = [Image.open(self.grayimgs[index]).convert("L") for index in
                range(start_index, stop_index + 1)]
        # print("start_img:{},stop_img:{}".format(self.grayimgs[start_index], self.grayimgs[stop_index]))
        dif_img = Image.fromarray(dif_img)        # img:PIL.Image   label:str
       
        if self.transform is not None:
            # imgs = [torch.squeeze(self.transform(Image.fromarray(img))) for img in imgs]
            imgs = [torch.squeeze(self.transform(img)) for img in imgs]
            dif_img = self.transform(dif_img)

        if self.labels[index]=="immobility":
            label = np.array("1").astype(np.int64)
            label = torch.from_numpy(label)
        else:
            label = np.array("0").astype(np.int64)
            label = torch.from_numpy(label)
        imgs = torch.stack(imgs)
        imgs = torch.cat([imgs,dif_img],dim=0)
        return imgs,label


class DatasetTest(Dataset):
    "TestDataset, derived from torch.utils.data.DataSet"

    def __init__(self, path, seq_len=15, transform=None) -> None:
        # self.txt_root = os.path.join(path,"data.txt")
        """seq_len 最好写ji数"""
        self.txt_root = path
        self.seq_len = seq_len
        with open(self.txt_root, 'r') as f:
            self.data = f.readlines()

        if not seq_len % 2:
            self.seq_len = seq_len + 1
        else:
            self.seq_len = seq_len

        grayimgs = []
        labels = []
        for line in self.data:
            line = line.rstrip()

            grayimg, label = line.split(' ')

            grayimgs.append(grayimg)
            labels.append(label)

        self.grayimgs = grayimgs
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # print("---------------INDEX ：{}-----------------------".format(index))

        start_index = index - self.seq_len // 2
        stop_index = index + self.seq_len // 2
        if self.seq_len > 750:
            print("----")
        if index % 750 >= (750 - self.seq_len // 2):
            stop_index = 750 - index % 750 - 1 + index
            start_index = stop_index - self.seq_len + 1
        if index % 750 < self.seq_len // 2:
            start_index = index - index % 750
            stop_index = start_index + self.seq_len - 1
        #
        # imgs = [np.array(cv2.imread(self.grayimgs[index], cv2.COLOR_BGR2GRAY)) for index in
        #         range(start_index, stop_index + 1)]
        imgs = [Image.open(self.grayimgs[index]).convert("L") for index in
                range(start_index, stop_index + 1)]

        # print("start_img:{},stop_img:{}".format(self.grayimgs[start_index],self.grayimgs[stop_index]))

        dif_img = diff_img(self.grayimgs, start_index, stop_index)
        dif_img = Image.fromarray(dif_img)        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:  # 目前他俩相同的 transform
            # imgs = [torch.squeeze(self.transform(Image.fromarray(img))) for img in imgs]
            imgs = [torch.squeeze(self.transform(img)) for img in imgs]
            dif_img = self.transform(dif_img)

        if self.labels[index] == "immobility":
            label = np.array("1").astype(np.int64)
            label = torch.from_numpy(label)
        else:
            label = np.array("0").astype(np.int64)
            label = torch.from_numpy(label)
        imgs = torch.stack(imgs)
        imgs = torch.cat([imgs, dif_img], dim=0)
        return imgs,label

def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
if __name__ =="__main__":
    setup_seed(2023)

    dataset = DatasetTrain(path="./train.txt",seq_len=3,transform=transform)
    # dataset = DatasetTest(path="./test_2250.txt", seq_len=3, transform=transform)

    ratio = 0.2
    data_loader= DataLoader(dataset, batch_size=8,
                              sampler=torch.utils.data.RandomSampler(dataset, num_samples=int(ratio * len(dataset))))
    count = 0
    for i,data in enumerate(data_loader):
        # print(len(data))
        gray,label =data
        count += len(label)
    # dataset.__getitem__(2243)
    print("一共采用{}图片".format(count))