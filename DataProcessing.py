import threading
import globalVariables
from process import Process
import numpy as np
from torchvision import transforms as T
import cv2

class DataProcessingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.localScoreMapBuffer = []
        self.process1finish = False
        self.process2finish = False
        self.transform_x = T.Compose([T.ToTensor()])
    def run(self):
        # Main program loop 
        process = Process()
        j = 0
        while True:
            if globalVariables.buffer1Ready == 1:
                j+=1
                self.process1finish = False
                # Process data from buffer 1
                patches = self.patchingImage(np.concatenate((globalVariables.superPatchBuffer[2*globalVariables.patchSize - globalVariables.borderSize: , :] , globalVariables.superPatchBuffer[:globalVariables.patchSize+globalVariables.borderSize,:]),axis=0),j)
                scoremap,self.process1finish = process.get_feature(patches)      
                globalVariables.buffer1Ready = 0  # Reset flag
                #conected patches and save it
                self.localScoreMapBuffer.append(np.concatenate(scoremap,axis=1))
            

            elif globalVariables.buffer2Ready == 1:
                # Process data from buffer 2
                j+=1
                self.process2finish = False
                patches = self.patchingImage(np.concatenate((globalVariables.superPatchBuffer[globalVariables.patchSize - globalVariables.borderSize: , :] , globalVariables.superPatchBuffer[:globalVariables.borderSize,:]),axis=0),j)
                scoremap,self.process2finish = process.get_feature(patches)
                globalVariables.buffer2Ready = 0  # Reset flag
                #conected patches and save it
                self.localScoreMapBuffer.append(np.concatenate(scoremap,axis=1))
                
            #sheet is finished
            if globalVariables.endOfSheet == 1 and self.process1finish and self.process2finish :
                    globalVariables.scoreMap.append(np.vstack(self.localScoreMapBuffer))
                    self.localScoreMapBuffer.clear()
                    globalVariables.endOfSheet = 0
                    globalVariables.resultReady = 1
                    print("number of superPatch",j)
    
    def patchingImage(self, image,j):
        """Extract patches from the image."""
        R, C = image.shape
        patches = []
        i = 0
        for c in range(globalVariables.borderSize, C - globalVariables.borderSize, globalVariables.patchSize):
              patch = image[:, c - globalVariables.borderSize:c + globalVariables.patchSize + globalVariables.borderSize].astype(np.uint8)
              patch = self.transform_x(patch)
              patches.append(patch)
              
              i += 1
        return np.stack(patches,axis=0)
