import torch
from torchvision.models import wide_resnet50_2
import numpy as np



class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        full_model = wide_resnet50_2(weights='DEFAULT', progress=True)
        state_dict = full_model.conv1.weight.data
        full_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        full_model.conv1.weight.data = state_dict[:, :1, :, :]
        base_model = torch.nn.Sequential(*list(full_model.children())[:6])
        self.base_mo = base_model 
        self.m = torch.nn.AvgPool2d(3, 1, 1)
        
        

    def forward(self, x):
        features = self.base_mo(x)
        features = self.m(features)
        patch_scores = self.calc_score(features)
        return patch_scores

    def calc_score(self, features):
        #calculate distance each feature from mean
        mean_features = torch.mean(features, dim=(2, 3))
        scores = torch.norm(features - mean_features[:, :, None, None], dim=1)
        return scores







