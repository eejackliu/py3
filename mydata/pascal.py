import torch
from torch.utils import data
from PIL import Image as image
import os
from skimage import io,transform
import numpy as np
class pascal_data(data.Dataset):
    root="data/VOCdevkit/VOC2012/"
    tmp = open("data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration = dict()
    paration["train"] = tmp
    tmp = open("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration["val"] = tmp
    tmp = open("data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt").readlines()
    tmp = [i.strip("\n") for i in tmp]
    paration["trainval"] = tmp
    def __init__(self,data_type="train",transorms=None):
        # self.label=labels
        # self.list_id=list_ids
        self.idx=self.paration[data_type]
        self.transorms=transorms
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, item):
        id=self.idx[item]
        # x=torch.load(self.root+"JPEGImages/"+id+".jpg")
        x=io.imread(self.root+"JPEGImages/"+id+".jpg")
        y=io.imread(self.root+"SegmentationClass/"+id+".png")
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
    def __call__(self, sample):
        source,target=sample['image'],sample['seg']
        image_h,image_w=source.shape[:2]
        if isinstance(self.output_size,tuple):
            h,w=self.output_size
        else:
            h=w=self.output_size
        top=np.random.randint(0,image_h-h)
        left=np.random.randint(0,image_w-w)
        new_source=source[top:top+h,left:left+w]
        new_target=target[top:top+h,left:left+w]
        return {"image":new_source,"seg":new_target}
class Normalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self, output):
        image=output['image']
        seg=output['seg']
        a=(image/255.0-self.mean)/self.std
        b=(seg/255.0-self.mean)/self.std
        return {"image":a,"seg":b}

class Totensor(object):
    def __call__(self, sample):
        image,target=sample['image'],sample['seg']
        image=np.transpose(image,axes=(2,0,1))
        target=np.transpose(target,axes=(2,0,1))

        return {'image':torch.from_numpy(image).float(),'seg':torch.from_numpy(target).float()}















