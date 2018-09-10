import torch
from torch.utils import data
from PIL import Image as image
import os
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
class pascal_data(data.Dataset):
    root="../torch/data/VOCdevkit/VOC2012/"
    tmp = open("../torch/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration = dict()
    paration["train"] = tmp
    tmp = open("../torch/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration["val"] = tmp
    tmp = open("../torch/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration["trainval"] = tmp
    def __init__(self,size,data_type="train",transorms=None):
        # self.label=labels
        # self.list_id=list_ids
        self.idx=self.paration[data_type]
        self.transorms=transorms
        if isinstance(size,tuple):
            self.h,self.w=size
        else:
            self.h=self.w=size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        id=self.idx[item]
        # x=torch.load(self.root+"JPEGImages/"+id+".jpg")
        x=io.imread(self.root+"JPEGImages/"+id+".jpg")
        y=io.imread(self.root+"SegmentationClass/"+id+".png")
        if x.shape[0]<=self.h or x.shape[1]<=self.w:
            return None

        if self.transorms:
            return self.transorms({"image":x,"seg":y})
        return {"image":x,"seg":y}
class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
    def __call__(self,sample):
        source,target=sample['image'],sample['seg']
        h,w=source.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w=sample*h/w,sample
            else:
                new_h,new_w=sample,sample*w/h
        else:
            new_h,new_w=self.output_size
        new_h,new_w=int(new_h),int(new_w)
        return {"image":transform.resize(source),'seg':transform.resize(target)}
class Randomcrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size=output_size
        if isinstance(self.output_size,tuple):
            self.h,self.w=self.output_size
        else:
            self.h=self.w=self.output_size

    def __call__(self, sample):
        source,target=sample['image'],sample['seg']
        image_h,image_w=source.shape[:2]
        if image_h<self.h:
            print("image h{} self h".format(image_h,self.h))
        if image_w<self.w:
            print("image w{} self w".format(image_w,self.w))
        top=np.random.randint(0,image_h- self.h)
        left=np.random.randint(0,image_w-self.w)
        new_source=source[top:top+self.h,left:left+self.w]
        new_target=target[top:top+self.h,left:left+self.w]
        return {"image":new_source,"seg":new_target}
class Normalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self, output):
        image=output['image']
        seg=output['seg']
        a=(image/255.0-self.mean)/self.std
        # b=(seg/255.0-self.mean)/self.std
        return {"image":a,"seg":seg}

class Totensor(object):
    def __init__(self):
        self.class_index=np.zeros(256**3)
        for i,j in enumerate(voc_colormap):
            tmp=(j[0]*256+j[1])*256+j[2]
            self.class_index[tmp]=i

    def __call__(self, sample):
        image,target=sample['image'],sample['seg']
        image=np.transpose(image,axes=(2,0,1))
        target=np.transpose(target,axes=(2,0,1))

        target=(target[0]*256+target[1])*256+target[2]
        target=self.class_index[target]
        return {'image':torch.from_numpy(image).float(),'seg':torch.from_numpy(target).long()}















