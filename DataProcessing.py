import threading
import globalVariables
from process import Process
import numpy as np
from torchvision import transforms as T

class DataProcessingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.localScoreMapBuffer = []
        globalVariables.resultReady = 0
        self.localScoreMapBufferPointer = 0
        self.transform_x = T.Compose([T.ToTensor()])
    def run(self):
        # Main program loop 
        process = Process()

        while True:
            #Check process buffer is ready
            if globalVariables.procBufferEmpty == 0:
                patches = self.patchingImage(globalVariables.processBuffer)
                scoremap = process.get_feature(patches)
                globalVariables.processBuffer.fill(globalVariables.backgroundGrayLevel)
                globalVariables.procBufferEmpty = 1
                self.localScoreMapBuffer.append(np.concatenate(scoremap,axis=1))
                
            #sheet is finished
            if globalVariables.endOfSheet == 1:
                    globalVariables.scoreMap.append(np.vstack(self.localScoreMapBuffer))
                    self.localScoreMapBuffer.clear()
                    globalVariables.endOfSheet = 0
                    globalVariables.resultReady = 1
    
    def patchingImage(self, image):
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
