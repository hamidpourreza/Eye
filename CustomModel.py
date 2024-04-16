import torch
from torchvision.models import wide_resnet50_2
import globalVariables



class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        full_model = wide_resnet50_2(weights='DEFAULT', progress=True)
        state_dict = full_model.conv1.weight.data
        full_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        full_model.conv1.weight.data = state_dict[:, :1, :, :]
        base_model = torch.nn.Sequential(*list(full_model.children())[:6])
        self.base_model = base_model 
        self.mean = torch.nn.AvgPool2d(3, 1, 1)
        

    def forward(self, x):
        features = self.base_model(x)
        features = self.mean(features)
        patch_scores = self.calcScore(features)
        return patch_scores

    def calcScore(self, features):
        #Calculate distance each feature from mean
        mean_features = torch.mean(features, dim=(2, 3))
        # Calculate the score for the last superPatch based on the second last mean superPatch
        if globalVariables.endOfSheet == 1 and globalVariables.bottomRow < globalVariables.limitationUsePreviousMean:
            if features.shape[0] == globalVariables.previousMean.shape[0]:
                scores = torch.norm(features - globalVariables.previousMean[:, :, None, None], dim=1)
            else:
                i = globalVariables.previousMean.shape[0] - features.shape[0]
                if globalVariables.correspondPatch:
                    scores = torch.norm(features - globalVariables.previousMean[:-1-i+1, :, None, None], dim=1)
                else:
                    scores = torch.norm(features - globalVariables.previousMean[i:, :, None, None], dim=1)
            globalVariables.correspondPatch = False
        elif globalVariables.usePreviousMean:
            mean_features[-1, :] = mean_features[-2, :]
            scores = torch.norm(features - mean_features[:, :, None, None], dim=1)
        else:
            scores = torch.norm(features - mean_features[:, :, None, None], dim=1)
        globalVariables.previousMean = mean_features

        return scores

 








