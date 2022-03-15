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
   # CONV 3X3 FILTROS2 STRIDE=1 PADDING=1\n",
   # BATCHNOMR\n",
   #RELU\n",
   # CONV 3X3 FLITROS3 STRIDE=1 PADDING=1\n",
   #MAXPOOL2D\n",
   def __init__(self,in_channels,kernel_size,filters, stride,padding):\n",
     filtro1,filtro2,filtro3 = filters\n",
     self.conv1= nn.Conv2d(in_channels=in_channels,out_channels=filtro1, kernel_size=kernel_size,stride=stride,padding=padding)\n",
     self.batchNomr1 =nn.BatchNorm2d(filtro1)\n",
     self.ReLU = nn.ReLU()\n",
     self.conv2= nn.Conv2d(in_channels=filtro1,out_channels=filtro2, kernel_size=kernel_size,stride=stride,padding=padding)\n",
     self.BatchNorm2 = nn.BatchNorm2d(filtro2)\n",
     self.conv3= nn.Conv2d(in_channels=filtro2,out_channels=filtro3, kernel_size=kernel_size,stride=stride,padding=padding)\n",
     self.BatchNorm3 = nn.BatchNorm2d(filtro3)\n",
     self.maxpool = nn.MaxPool2d(2,1)\n",
   def fordward(self,x):\n",
     x = self.conv1(x)\n",
     x= self.BatchNorm1(x)\n",
     x= self.ReLU(x)\n",
     x = self.conv2(x)\n",
     x= self.BatchNorm2(x)\n",
     x= self.ReLU(x)\n",
     x = self.conv3(x)\n",
     x= self.BatchNorm3(x)\n",
     x= self.ReLU(x)\n",
     x =self.maxpool(x)\n",
        "\n",
        "    return x\n",
        "\n",
        "\n",
        "@BACKBONE_REGISTRY.register()\n",
        "class modelo1a(Backbone):\n",
        "\n",
        "  def __init__(self, cfg, input_shape):\n",
        "    super().__init__()\n",
        "    self.conv_block = conv_block()\n",
        "    self.Zpadding = nn.ZeroPad2d((3,3))\n",
        "    self.conv1 = nn.Conv2d(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)\n",
        "    self.bn = nn.BatchNorm1d(64)\n",
        "    self.relu = nn.ReLU()\n",
        "    self.maxpool = nn.MaxUnpool2d((3, 3), strides=(2, 2), padding=\"same\")\n",
        "\n",
        "  def forward(self, image):\n",
        "    x = self.Zpadding(image)\n",
        "    x = self.conv1(x)\n",
        "    x = self.bn(x)\n",
        "    x = self.relu(x)\n",
        "    C1 = x = self.maxpool(x)\n",
        "    # Stage 2\n",
        "                     #self,in_channels,kernel_size,filters, stride,padding\n",
        "    x = self.conv_block(x, 3, [64, 64, 128],(1, 1),(1,1))\n",
        "    C2=x\n",
        "    \n",
        "    # Stage 3\n",
        "    x = self.conv_block(x, 3, [128, 128, 256], (1,1),(1,1))\n",
        "    C3=x\n",
        "    # Stage 4\n",
        "    x = self.conv_block(x, 3, [256, 256, 512],(1,1),(1,1) )\n",
        "    C4 = x\n",
        "    # Stage 5\n",
        "    x = self.conv_block(x, 3, [512, 512, 1024],(1,1),(1,1))\n",
        "        \n",
        "    C5 = x\n",
        "    return [C1, C2, C3, C4, C5]\n",
        "\n",
        "  #def output_shape(self):\n",
        "   # return {\"conv1\": ShapeSpec(channels=64, stride=16)}\n",
        "\n",
        "cfg = ...   # read a config\n",
        "cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # or set it in the config file\n",
        "model = build_backbone(cfg)  # it will find `ToyBackbone` defined above \n"
      ],
      "metadata": {
        "id": "xUfBwyAsSrmq"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
