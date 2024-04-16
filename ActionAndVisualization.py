import threading
import globalVariables
import numpy as np
import cv2
import os

class ActionAndVisualizationThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.numberSheet = 0
        #CLAHE parameter
        self.clip_limit=2.0, 
        self.grid_size=(8, 8)
    

    def run(self):
        while True:
            # Wait for data to be processed and ready for action or visualization
            if globalVariables.resultReady == 1:
                scoreMap = globalVariables.scoreMap.pop(0)
                imageMask = globalVariables.imageMask.pop(0)
                rawImage = globalVariables.rawImage.pop(0)
                #calculate enhanced Image
                enhancedImage = self.CLAHE(rawImage)
               
                self.saveImage("imageMask.png", imageMask.astype(np.uint8))
                self.saveImage("rawImage.png", rawImage)
                self.saveImage("enhancedImage.png", enhancedImage)
                scoreMap0 = self.normalization(scoreMap)
                self.saveImage("scoreMap.png", scoreMap0.astype(np.uint8))

                #applay imageMask to scoreMap
                imageMask = imageMask[:scoreMap.shape[0], :]
                result = np.where(imageMask == 0, 0, scoreMap)

                #Calculate defectMap
                defectMap = np.where(result >globalVariables.threshold, 255, 0)
                self.saveImage("defectMap.png", defectMap.astype(np.uint8))
                result = self.normalization(result)
                self.saveImage("result.png", result.astype(np.uint8))

                #Overlay defectMap on rawImage
                overlay = rawImage[:defectMap.shape[0], :]
                # overlay[defectMap != 255] = 0

                # Find contours
                contours, _ = cv2.findContours(defectMap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  
                
                self.saveImage("overlay.png", overlay.astype(np.uint8))

                #Overlay defectMap on enhancedImage
                overlay = enhancedImage[:defectMap.shape[0], :]
                # overlay[defectMap != 255] = 0

                # Draw contours
                cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  
                
                self.saveImage("enhancedImageOverlay.png", overlay.astype(np.uint8))
            
                self.numberSheet += 1
                globalVariables.resultReady = 0
                print("finish")


    def normalization(self,image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = ((image - min_val)/(max_val - min_val))*255
        return normalized_image

    def CLAHE(self,image, clip_limit=2.0, grid_size=(8, 8)):
        # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        # Apply CLAHE to the grayscale image
        enhanced_image = clahe.apply(image)
        
        return enhanced_image
    
    def saveImage(self, name, image):
        path = globalVariables.outputPath+"sheet%d/"%self.numberSheet
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+name, image)




   