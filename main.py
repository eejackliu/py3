# %%
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.models as model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# #%%
# import torch.optim as optim
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1=nn.Conv2d(3,6,3,padding=1)
#         self.pool1=nn.MaxPool2d(2,2)
#         self.conv2=nn.Conv2d(6,16,3,padding=1)
#         self.pool2=nn.MaxPool2d(2,2)
#         self.fc1=nn.Linear(16*8*8,120)
#         self.fc2=nn.Linear(120,10)
#     def forward(self, *input):
#         x=self.pool1(self.conv1(input))
#         x=self.pool2(self.conv2(x))
#         x=x.view(-1,16*8*8)
#         x=f.relu(self.fc1(x))
#         x=f.relu(self.fc2(x))
#         return x
# #%%
#
# # para=list(vgg.children())[1]
# # print(para)
# #%%
#
# #%%
# def imshow(img):
#     img=img*0.5+0.5
#     npimg=img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0))) # thanspose from (CHW) to(HWC)
# tran_conv=nn.ConvTranspose2d(3,3,kernel_size=4,padding=1,stride=2)
# tran_conv.weight=torch.nn.Parameter(torch.from_numpy(bilinear_kernel(3,3,4)))
# out=tran_conv(a)
# imshow(out.detach()[0])
# plt.show()





#%%
from mydata import pascal
import matplotlib.pyplot as plt
from torchvision import  transforms
import numpy as np
import torch.optim as optim
import torchvision.models as model
import torch.nn  as nn
import torch.nn.functional as f
import torch
import torchvision
from skimage import io
voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
transform=transforms.Compose([pascal.Randomcrop((224,224)),pascal.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),pascal.Totensor()])
trainset=pascal.pascal_data((224,224),"train",transform)
def my_collate_fn(batch):
    item=list(filter(lambda x:x is  not None,batch))
    # print(batch)
    return torch.utils.data.dataloader.default_collate(item)
a=torch.utils.data.DataLoader(trainset,4,shuffle=True,collate_fn=my_collate_fn)
# train_data=iter(a)
vgg=model.vgg16(pretrained=True)
class Fcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.numclass=21
        self.conv=list(vgg.children())[0]
        self.conv1=nn.Conv2d(512,self.numclass,1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.tran_conv=nn.ConvTranspose2d(self.numclass,self.numclass,64,32,16)
        self.tran_conv.weight=torch.nn.Parameter(torch.from_numpy(self.bilinear_kernel(21,21,64)))
    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size),
            dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return np.array(weight)
    def forward(self, input):
        x=self.conv(input)
        x=self.conv1(x)
        x=self.tran_conv(x)
        return x
def picture(tmp):
    mean,std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
    fig,ax=plt.subplots(2,4)
    for n,i in enumerate(ax):
        for m,j in enumerate(i):
            if n:
                j.imshow(np.transpose(tmp[m].numpy(),axes=(1,2,0)))
                # j.imshow(np.transpose(tmp['seg'][m].numpy(),axes=(1,2,0))*std+mean)

            else:
                # j.plot(a[m]['image'])
                j.imshow(np.transpose(tmp[m].numpy(),axes=(1,2,0)))
                # j.imshow(np.transpose(tmp['image'][m].numpy(),axes=(1,2,0))*std+mean)
    plt.show()
fcn=Fcn()
out=nn.Softmax2d()
criterion=nn.CrossEntropyLoss()
optimize=optim.SGD(fcn.parameters(),lr=0.001,momentum=0.9)
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# fcn=fcn.to(device)
for i in range(1):
    # tmp=next(train_data)
    for m in  a:
        # input_data,lable_data=tmp['image'],tmp['seg']
        input_data,lable_data=m['image'],m['seg']
        print(input_data.shape)
        # input_data,lable_data=input_data.to(device),lable_data.to(device)
        optimize.zero_grad()
        output=fcn(input_data)
        # print(output.size())
        # print(lable_data.size())
        loss=f.softmax(output,dim=1)
        loss=criterion(loss,lable_data)
        print (loss)
        loss.backward()
        optimize.step()
        break
tmp=next(train_data)
test=fcn(tmp['image'])


#%%
# np.set_printoptions(threshold=np.inf)

# colormap2label = np.zeros(48**3)
# for i, cm in enumerate(voc_colormap):
#     colormap2label[(cm[0] * 16 + cm[1]) * 16 + cm[2]] = i
# print(colormap2label)
# import matplotlib.pyplot as plt
#
# from skimage import io,transform
# root="data/VOCdevkit/VOC2012/"
# data= np.zeros(256 ** 3)
#
# #
# x=io.imread(root+"JPEGImages/"+'2007_000042'+".jpg")
# y=io.imread(root+"SegmentationClass/"+'2007_000042'+".png")
# x=y.astype('int32')
# mask=dict()
# for i,j in enumerate(voc_colormap):
#     mask[i]=(j[0]*256+j[1])*256+j[2]
# x=x[50:250,50:100]
# a=(x[:,:,0]*256+x[:,:,1])*256+x[:,:,2]
# dv=np.array(list(mask.values()))
# dk=np.array(list(mask.keys()))
# # out=np.where(a[:,:,None]==dv)
# out=np.isin(a,dv)
# d=a[out]

# color2map=np.zeros(256**3)
# for i,j in enumerate(voc_colormap):
#     color2map[(j[0]*256+j[1])*256+j[2]]=i
# a=(x[:,:,0]*256+x[:,:,1])*256+x[:,:,2]
# out=color2map[a]
#
# print(a.max())
# p=color2map[a]
# print (p.max())
# print(p[100:150,50:100])
# # plt.imshow(img_x)
# # plt.subplot(2,2,2)
# # plt.imshow(img_y)
# # plt.show()
# print(img_y)
# print(img_y[:,:,0])
# tmp=img_y[:,:,0]
# plt.imshow(img_x,cmap='gray')
# plt.show()
# def __init__(self,data_type="train",transorms=None):
#     # self.label=labels
#     # self.list_id=list_ids
#     self.idx=self.paration[data_type]
#     self.transorms=transorms
# def __len__(self):
#     return len(self.idx)
# def __getitem__(self, item):
#     id=self.idx[item]
#     # x=torch.load(self.root+"JPEGImages/"+id+".jpg")
#     x=io.imread(self.root+"JPEGImages/"+id+".jpg")
#     y=io.imread(self.root+"SegmentationClass/"+id+".png")
#     if self.transorms:
#         return self.transorms({"image":x,"seg":y})
#     return {"image":x,"seg":y}
# import torchvision.models as model
#
# # tmp=next(train_data)
# # input_data,lable_data=tmp['image'],tmp['seg']
# # b=vgg(input_data)
# # print(b.Size())
# resnet=model.resnet18(pretrained=True )
# print (*list(resnet.children()))
# a=resnet.fc.in_features
# b=resnet.state_dict()
# torch.sa
# b=torchvision.utils.make_grid(a['seg'])
# c=torchvision.utils.make_grid(a['image'])
# plt.subplot(221)
# plt.show()