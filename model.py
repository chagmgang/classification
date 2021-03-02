import torch
import torchvision

import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()
        
        #### torchvision에서 resnet50을 가져와 모델을 만들어보세요

    def forward(self, image):
        logit = self.model(image)
        return logit

if __name__ == '__main__':
    net = Model(num_class=100)

    ones = torch.ones([3, 3, 32, 32])
    net(ones)
