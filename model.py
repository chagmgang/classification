import torch
import torchvision

import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feature, num_class)

    def forward(self, image):
        logit = self.model(image)
        return logit

if __name__ == '__main__':
    net = Model(num_class=100)

    ones = torch.ones([3, 3, 32, 32])
    net(ones)
