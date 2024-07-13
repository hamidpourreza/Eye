import threading
import time
import os 
import cv2
import numpy as np
import globalVariables
from pypylon import pylon
import scipy

# Data Acquisition Thread
class DataAcquisitionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.bufferPointer = 0
        self.firstPatch = 1
        self.cameraImagePointer = 0
        self.minBackgroundLevel = 235
        #Where super patch started
        self.startOfBuffer = 0    
        self.localCameraImage = []
        self.localMask = []
        self.localRawImage = []
        self.localImage = []
        #Grab image from camera or simulator
        self.grabFromCamera = False

        globalVariables.sheetDetect = 0
        globalVariables.cameraDetect = 0
        globalVariables.nonRealtime = 0
        globalVariables.patchCounter = 0
        globalVariables.endOfSheet = 0
        globalVariables.lineReceived = 0
        globalVariables.procBufferEmpty = 1
        
        
        
        

    def run(self):
        #Camera setting
        if self.grabFromCamera:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()
            camera.Width.Value = globalVariables.sensorSize
            camera.Height.Value = globalVariables.linePackSize
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        else:
            #Image from simulator
            data = self.loadDatasetFolder()
            index = 0
        #Fill top border of first super patch
        globalVariables.dataBuffer[:globalVariables.borderSize,:] = globalVariables.backgroundGrayLevel
        self.bufferPointer = globalVariables.borderSize
        #Super patch 0 start from index 0
        self.startOfBuffer = 0
        while True:
                # Access the image data
                if self.grabFromCamera:
                    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    image = grabResult.GetArray()
                    grabResult.Release()
                else:
                    if index >= len(data):
                         break
                    image = cv2.imread(data[index],cv2.IMREAD_GRAYSCALE)
                    index += 1
      
                self.storeCameraImage(image)
                globalVariables.lineReceived += globalVariables.linePackSize
                    
                if self.allLinesAreEmpty(image):
                    if globalVariables.sheetDetect != 1:
                        continue
                    else:
                        if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                            if globalVariables.procBufferEmpty == 0:
                                print("non real time")
                                globalVariables.nonRealtime = 1
                                continue
                            else:
                                self.handleProcessingBuffer(image)
                        else:
                            self.SavePackLine(image)
                            if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                                if globalVariables.procBufferEmpty == 0:
                                    print("non real time")
                                    globalVariables.nonRealtime = 1
                                    continue
                                else:
                                    self.handleProcessingBuffer(image)
                                
                            else:
                                continue
                else:
                    globalVariables.sheetDetect = 1
                    if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                        if globalVariables.procBufferEmpty == 0:
                            print("non real time")
                            globalVariables.nonRealtime = 1
                            continue
                        else:
                            self.handleProcessingBuffer(image)
                    else:
                        self.SavePackLine(image)
                        if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                            if globalVariables.procBufferEmpty == 0:
                                print("non real time")
                                globalVariables.nonRealtime = 1
                                continue
                            else:
                                self.handleProcessingBuffer(image)

            


    # Functions 
                    
    def SavePackLine(self, image):
        #Store raw image and mask
        self.storeRawImage(image)
        self.createAndStoreMask(image)
        #Save packLine in data buffer
        start = self.bufferPointer% globalVariables.bufferSize
        globalVariables.dataBuffer[start:start+globalVariables.linePackSize//2,
                                    globalVariables.borderSize:globalVariables.sensorSize+globalVariables.borderSize] = image[:globalVariables.linePackSize//2, :]
        self.bufferPointer = (self.bufferPointer + globalVariables.linePackSize//2)
        start = self.bufferPointer% globalVariables.bufferSize
        globalVariables.dataBuffer[start:start+globalVariables.linePackSize//2,
                                    globalVariables.borderSize:globalVariables.sensorSize+globalVariables.borderSize] = image[globalVariables.linePackSize//2:, :]
        self.bufferPointer = (self.bufferPointer + globalVariables.linePackSize//2)
        


    def dataBufferToProcessBuffer(self):
        #Process Buffer is empty
        #Compute start and end of super patch in data buffer
        s = self.startOfBuffer % globalVariables.bufferSize
        end = globalVariables.superPatchSize - (globalVariables.bufferSize - s)
        if s == 0:
            globalVariables.processBuffer[:, :] = globalVariables.dataBuffer[:globalVariables.superPatchSize, :]
        else:
            globalVariables.processBuffer[:, :] = np.vstack((globalVariables.dataBuffer[s:, :],globalVariables.dataBuffer[:end, :]))
        #Fill triangles resulting from rotation and padding around image with sheet border value
        #globalVariables.padding = True
        s = time.time()
        self.fillTriangles()
        print("fill triangles time:", time.time() - s)
        s = time.time()
        self.paddingLeftRightTopButtom()
        print("padding time:", time.time() - s)
        globalVariables.procBufferEmpty = 0 #set process Buffer is full
        globalVariables.patchCounter += 1
        self.startOfBuffer = self.startOfBuffer + globalVariables.patchSize

        

    def handleProcessingBuffer(self, image):
        self.dataBufferToProcessBuffer()
        if globalVariables.patchCounter >= globalVariables.maxSheetLength:
            self.endOfSheet()
        elif np.all(image[-1, :] > self.minBackgroundLevel):
            globalVariables.dataBuffer[:globalVariables.borderSize, :] = globalVariables.backgroundGrayLevel
            self.bufferPointer = globalVariables.borderSize
            self.startOfBuffer = 0
            globalVariables.sheetDetect = 0
            self.endOfSheet()



    def endOfSheet(self):
        #Save all image and reset buffers
        globalVariables.patchCounter = 0
        globalVariables.rawImage.append(np.vstack(self.localRawImage))
        globalVariables.imageMask.append(np.vstack(self.localMask))
        globalVariables.cameraImage.append(np.vstack(self.localCameraImage))
        print("complete reading image")
        self.localRawImage.clear()
        self.localMask.clear()
        self.localCameraImage.clear()
        globalVariables.endOfSheet = 1   

    def loadDatasetFolder(self):
        x = []
        image_dir = globalVariables.inputPath
        files = os.listdir(image_dir)
        files.sort(key=lambda x: int(x.split('.')[0]))
        for image_type in files:
            #Load images
            image_type_dir = os.path.join(image_dir, image_type)
            x.append(image_type_dir)
        return list(x)       

    def fillTriangles(self):
        _, mask = cv2.threshold(globalVariables.processBuffer, self.minBackgroundLevel, globalVariables.backgroundGrayLevel, cv2.THRESH_BINARY_INV)
        kernel = np.ones((11,11), np.uint8)
        # Erode the binary image
        mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(contours) <= 0:
                return
        max_contour = max(contours, key=cv2.contourArea)
        # Find the bounding rectangle of the sheet
        x, y, w, h = cv2.boundingRect(max_contour)
        sub_mask = mask[y:y+h, x:x+w]
        nearest_neighbor = scipy.ndimage.distance_transform_cdt(sub_mask == 0,metric='taxicab' ,return_distances=False, return_indices=True) 
        nn_rows, nn_cols = nearest_neighbor
        sub_img = globalVariables.processBuffer[y:y+h, x:x+w]
        # Fill triangles with non-zero nearest neighbor 
        sub_img[sub_mask == 0] = sub_img[nn_rows[sub_mask == 0], nn_cols[sub_mask == 0]]

        # Place the processed sub-region back into the original image
        globalVariables.processBuffer[y:y+h, x:x+w] = sub_img

          
    
    def paddingLeftRightTopButtom(self):
        R, C = globalVariables.processBuffer.shape
        _, mask = cv2.threshold(globalVariables.processBuffer, self.minBackgroundLevel, globalVariables.backgroundGrayLevel, cv2.THRESH_BINARY_INV)
        kernel = np.ones((11,11), np.uint8)
        # Erode the binary image
        mask = cv2.erode(mask, kernel, iterations=1)

        #Padding left and right
        #Find the minimum and maximum non-zero indices along each row
        min_indices = np.argmax(mask, axis=1)
        max_indices = C - np.argmax(np.flip(mask, axis=1), axis=1) - 1
        
        #Find left and right columns to remove the redundant background
        if min(min_indices) != 0 and globalVariables.endOfSheet != 1:
            globalVariables.leftColumn = min(min_indices)
            globalVariables.rightColumn = max(max_indices)
        else:
            globalVariables.correspondPatch = False
            pre = globalVariables.leftColumn
            globalVariables.leftColumn = np.argmax(mask.any(axis=0))
            globalVariables.rightColumn = mask.shape[1] - np.argmax(mask[:, ::-1].any(axis=0)) - 1
            if globalVariables.leftColumn - pre < globalVariables.limitationUsePreviousMean:
                 globalVariables.correspondPatch = True
                 
                 
    
        #Update processBuffer for padding left and right
        for r in range(R):
            min_val = globalVariables.processBuffer[r, min_indices[r]]
            max_val = globalVariables.processBuffer[r, max_indices[r]]
            globalVariables.processBuffer[r, :min_indices[r]] = min_val
            globalVariables.processBuffer[r, max_indices[r]+1:] = max_val
       
        #Padding top and bottom
        _, mask = cv2.threshold(globalVariables.processBuffer, self.minBackgroundLevel, globalVariables.backgroundGrayLevel, cv2.THRESH_BINARY_INV)
        #Erode the binary image
        mask = cv2.erode(mask, kernel, iterations=1)
        ind = np.where(mask[:, 0] != 0)
        if len(ind[0]) > 0:
            minimum,maximum = min(ind[0]),max(ind[0])
            globalVariables.processBuffer[globalVariables.leftColumn:minimum, :] = globalVariables.processBuffer[minimum, :]
            globalVariables.processBuffer[maximum: globalVariables.rightColumn, :] = globalVariables.processBuffer[maximum, :] 
            globalVariables.bottomRow = maximum


    def createAndStoreMask(self,image):
        _, thresholded = cv2.threshold(image, self.minBackgroundLevel, globalVariables.backgroundGrayLevel,cv2.THRESH_BINARY_INV )
        self.localMask.append(thresholded)
       

    def storeRawImage(self,image):
        self.localRawImage.append(image)


    def storeCameraImage(self,image):
        self.localCameraImage.append(image)
        self.cameraImagePointer +=  globalVariables.linePackSize
        if self.cameraImagePointer == globalVariables.cameraImageLength:
            image = np.vstack(self.localCameraImage) 
            globalVariables.cameraImage.append(image)
            self.cameraImagePointer = 0
            self.localCameraImage.clear()

    def allLinesAreEmpty(self,image):
        _, th = cv2.threshold(image, self.minBackgroundLevel, globalVariables.backgroundGrayLevel,cv2.THRESH_BINARY_INV )
        if  np.any(th!= 0):
            return False
        else:
            return True
        
  
        



