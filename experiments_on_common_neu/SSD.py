import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import d2l

#分类输出层
def cls_predictor(num_imputs,a,q):
    return nn.Conv2d(num_imputs,a*(q+1),kernel_size=3,padding=1)

#边界框预测层,每个锚框设置4个偏移量
def bbox_predictor(num_inputs,a):
    return nn.Conv2d(num_inputs,a*4,kernel_size=3,padding=1)

#举例
def forward(x,block):
    return block(x)
Y1=forward(torch.zeros(2,8,20,20),cls_predictor(8,5,10))
Y2=forward(torch.zeros(2,16,10,10),cls_predictor(16,3,10))
print(Y1.shape)
print(Y2.shape)

#通道维包含中心相同的锚框的预测结果。我们首先将通道维移到最后一维。
# 因为不同尺度下批量大小仍保持不变，我们可以将预测结果转成二维的（批量大小，高宽*通道数）的格式，以方便之后在维度
#上的连结。
def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)
def concat_pred(pred):
    return torch.cat([flatten_pred(p) for p in pred],dim=1)
print(concat_pred([Y1, Y2]).shape)

#高和宽减半
#将输入特征图高宽减半，类似于VGG设计，每个高宽减半由两个padding=1的3*3conv2d，和两个stribe 2的2*2maxpooling
def down(in_channel,out_channel):
    blk=[]
    for i in range(2):
        blk.append(nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1))
        blk.append(nn.BatchNorm2d(out_channel))
        blk.append(nn.ReLU())
        in_channel=out_channel
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
#构建的高和宽减半块会更改输入通道的数量
print(forward(torch.zeros((2, 3, 20, 20)), down(3, 10)).shape)


#基本网络块，将通道数依次翻倍
def base_net():
    blk=[]
    num_filters=[3,16,32,64]
    for i in range(len(num_filters)-1):
        blk.append(down(num_filters[i],num_filters[i+1]))
    return nn.Sequential(*blk)
#input 256*256
print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

#完整的单发多框检测模型由五个模块组成。每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。
# 在这五个模块中，第一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。

def get(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down(128, 128)
    return blk

def forward_flk(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class Tinyssd(nn.Module):
    def __init__(self,num_classes,**kwards):
        super(Tinyssd,self).__init__(**kwards)

        self.num_classes=num_classes
        index_to_channels=[64,128,128,128,128]
        for i in range(5):
            setattr(self, f'blk_{i}', get(i))
            setattr(self, f'cls_{i}', cls_predictor(index_to_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(index_to_channels[i],
                                                      num_anchors))

    def forward(self,X):
        anchors,cls_predict,bbox_predict=[None]*5,[None]*5,[None]*5
        for i in range(5):
            X,anchors[i],cls_predict[i],bbox_predict[i]=forward_flk(X,getattr(self,f'blk_{i}'),sizes[i],ratios[i],
                                                           getattr(self,f'cls_{i}'),getattr(self,f'bbox_{i}'))
        anchors=torch.cat(anchors,dim=1)#预测类别数就是通道数，对列联结，（批量大小，高宽*通道数）
        cls_predict=concat_pred(cls_predict)
        cls_predict=cls_predict.reshape(cls_predict.shape[0],-1,self.num_classes+1)
        bbox_predict=concat_pred(bbox_predict)
        return anchors,cls_predict,bbox_predict

net = Tinyssd(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)




