from abc import ABCMeta, abstractmethod
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone\n
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
__all__ = [\"Backbone\"]
           
class conv_block():
   # CONV 3X3  FLITROS1 STRIDE=1 PADDING=1
   # BATCHNORM
   #RELU
   # CONV 3X3 FILTROS2 STRIDE=1 PADDING=1
   # BATCHNOMR
   #RELU
   # CONV 3X3 FLITROS3 STRIDE=1 PADDING=1
   #MAXPOOL2D
  def __init__(self,in_channels,kernel_size,filters, stride,padding):
    filtro1,filtro2,filtro3 = filters
    self.conv1= nn.Conv2d(in_channels=in_channels,out_channels=filtro1, kernel_size=kernel_size,stride=stride,padding=padding)
    self.batchNomr1 =nn.BatchNorm2d(filtro1)
    self.ReLU = nn.ReLU()
    self.conv2= nn.Conv2d(in_channels=filtro1,out_channels=filtro2, kernel_size=kernel_size,stride=stride,padding=padding)
    self.BatchNorm2 = nn.BatchNorm2d(filtro2)
    self.conv3= nn.Conv2d(in_channels=filtro2,out_channels=filtro3, kernel_size=kernel_size,stride=stride,padding=padding)
    self.BatchNorm3 = nn.BatchNorm2d(filtro3)
    self.maxpool = nn.MaxPool2d(2,1)
  def fordward(self,x):
     x = self.conv1(x)
     x= self.BatchNorm1(x)
     x= self.ReLU(x)
     x = self.conv2(x)
     x= self.BatchNorm2(x)
     x= self.ReLU(x)
     x = self.conv3(x)
     x= self.BatchNorm3(x)
     x= self.ReLU(x)
     x =self.maxpool(x)

     return x
@BACKBONE_REGISTRY.register()
class modelo1a(Backbone):

 def __init__(self, cfg, input_shape):
   super().__init__()
   self.conv_block = conv_block()
   self.Zpadding = nn.ZeroPad2d((3,3))
   self.conv1 = nn.Conv2d(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)
   self.bn = nn.BatchNorm1d(64)
   self.relu = nn.ReLU()
   self.maxpool = nn.MaxUnpool2d((3, 3), strides=(2, 2), padding=\"same\")

 def forward(self, image):
   x = self.Zpadding(image)
   x = self.conv1(x)
   x = self.bn(x)
   x = self.relu(x)
   C1 = x = self.maxpool(x)
   # Stage 2
   #self,in_channels,kernel_size,filters, stride,padding
   x = self.conv_block(x, 3, [64, 64, 128],(1, 1),(1,1))
   C2=x

# Stage 3
   x = self.conv_block(x, 3, [128, 128, 256], (1,1),(1,1))
   C3=x
   # Stage 4
   x = self.conv_block(x, 3, [256, 256, 512],(1,1),(1,1) )
   C4 = x
   # Stage 5
   x = self.conv_block(x, 3, [512, 512, 1024],(1,1),(1,1))

   C5 = x
   return [C1, C2, C3, C4, C5]

 #def output_shape(self):
  # return {\"conv1\": ShapeSpec(channels=64, stride=16)}
cfg = ...   # read a config
cfg.MODEL.BACKBONE.NAME = 'modelo1a'   # or set it in the config file
model = build_backbone(cfg)  # it will find `ToyBackbone` defined above 

