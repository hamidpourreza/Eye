import torch
from custom_model import CustomModel
import torch.nn.functional as F
import globalVariables
import numpy as np
import torchvision
import cv2


class Process: 
  def __init__(self, cut_surrounding=6):
     
     self.cut_surrounding = cut_surrounding
     

  def get_feature(self,images):
    #set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #send data to gpu
    image_patches = torch.tensor(images, device=device, dtype=torch.float32)
    #initial model
    model = CustomModel()
    model.to(device)
    model.eval()
    #get score map
    with torch.no_grad():
        heatMap = model(image_patches)
    #cut border
    heatMap = heatMap[:, self.cut_surrounding:heatMap.shape[1]-self.cut_surrounding,
                                self.cut_surrounding:heatMap.shape[2] - self.cut_surrounding]
    
    imgshape= (globalVariables.patchSize,globalVariables.patchSize)
    heatMap1 = torch.zeros((heatMap.shape[0],globalVariables.patchSize,globalVariables.patchSize))
    
    #rezise to Original size
    for i in range(heatMap.shape[0]):
      heatMap1[i, :, :] = torchvision.transforms.functional.resize(heatMap[i, : , :].unsqueeze(0).unsqueeze(0),imgshape)
    return heatMap1.cpu().detach().numpy(), True