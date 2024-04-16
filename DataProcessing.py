import threading
import globalVariables
import numpy as np
from torchvision import transforms as T
import cv2
from CustomModel import CustomModel
import torch


class DataProcessingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.localScoreMapBuffer = []
        globalVariables.resultReady = 0
        self.localScoreMapBufferPointer = 0
        self.ready = False
        self.transform_x = T.Compose([T.ToTensor()])

    def run(self):
        # Main program loop 
        while True:
            #Check process buffer is ready
            self.ready = False
            if globalVariables.procBufferEmpty == 0:
                patches = self.patchingImage()
                heatMap, self.ready = self.getFeature(patches)

                globalVariables.processBuffer.fill(globalVariables.backgroundGrayLevel)
                globalVariables.procBufferEmpty = 1
                
 
                # Resize heatMap to the original size
                imgshape = (globalVariables.superPatchSize, globalVariables.superPatchSize)
                scoremap = np.zeros((heatMap.shape[0], globalVariables.superPatchSize, globalVariables.superPatchSize))
              
                for i in range(heatMap.shape[0]):
                    scoremap[i, :, :] = cv2.resize(heatMap[i, :, :], imgshape)
                
                # Remove border from scoremap
                scoremap = scoremap[:, globalVariables.borderSize:-globalVariables.borderSize,
                                    globalVariables.borderSize:-globalVariables.borderSize]
          
                # Concatenate scoremap along the correct 
                end = globalVariables.leftColumn + (scoremap.shape[-1]*scoremap.shape[0]) - globalVariables.borderSize
                start = globalVariables.leftColumn - globalVariables.borderSize
                scoremap0 = np.zeros((globalVariables.patchSize, globalVariables.sensorSize))
                if end > globalVariables.sensorSize:
                    scoremap0[:, start:] = np.concatenate(scoremap, axis=1)[:, :globalVariables.sensorSize-start]
                else:
                    scoremap0[:, start:end] = np.concatenate(scoremap, axis=1)
              
                self.localScoreMapBuffer.append(scoremap0)
                
            #Sheet is finished
            if globalVariables.endOfSheet == 1 and self.ready:
                    globalVariables.scoreMap.append(np.vstack(self.localScoreMapBuffer))
                    self.localScoreMapBuffer.clear()
                    globalVariables.endOfSheet = 0
                    globalVariables.resultReady = 1
                    
    
    def patchingImage(self):
        """Extract patches from the image."""
        patches = []
        globalVariables.usePreviousMean = False
        
        for c in range(globalVariables.leftColumn, globalVariables.rightColumn, globalVariables.patchSize):
              patch = globalVariables.processBuffer[:, c - globalVariables.borderSize:c + globalVariables.patchSize + globalVariables.borderSize]
              if patch.shape[1] < globalVariables.superPatchSize:
                lastPatch = np.ones((globalVariables.superPatchSize, globalVariables.superPatchSize)) * patch[:, -1]
                lastPatch[:, :patch.shape[-1]] = patch
                patch = lastPatch
                globalVariables.usePreviousMean = True
         
              patch = self.transform_x(patch.astype(np.uint8))
              patches.append(patch)

        #Background in patch is more than sheet piece
        if globalVariables.rightColumn - c < globalVariables.limitationUsePreviousMean:
            globalVariables.usePreviousMean = True
         
        return torch.stack(patches,axis=0)
    

    def getFeature(self, patches):
        """Get feature scores for image patches."""
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Send data to GPU
        image_patches = patches.to(device=device, dtype=torch.float32)
        
        # Initialize and evaluate the model
        model = CustomModel()
        model.to(device)
        model.eval()

        with torch.no_grad():
            heatMap = model(image_patches)

        return heatMap.cpu().detach().numpy(),  True
    
    
    
