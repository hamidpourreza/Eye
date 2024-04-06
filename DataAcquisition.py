import threading
import time
import os 
import cv2
import numpy as np
import globalVariables


# Data Acquisition Thread
class DataAcquisitionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.bufferPointer = 0
        self.firstPatch = 1
        self.cameraImagePointer = 0
        self.minBackgroundLevel = 235
        #where super patch started
        self.startOfBuffer = 0    
        self.localCameraImage = []
        self.localMask = []
        self.localRawImage = []
        self.localImage = []

        globalVariables.sheetDetect = 0
        globalVariables.cameraDetect = 0
        globalVariables.nonRealtime = 0
        globalVariables.patchCounter = 0
        globalVariables.endOfSheet = 0
        globalVariables.lineReceived = 0
        globalVariables.procBufferEmpty = 1
        self.i = 0

    def run(self):
        data = self.load_dataset_folder()
        #fill top border of first super patch
        globalVariables.dataBuffer[:globalVariables.borderSize,:] = globalVariables.backgroundGrayLevel
        self.bufferPointer = globalVariables.borderSize
        #super patch 0 start from index 0
        self.startOfBuffer = 0

        for x in data:
            #get linepack
            image = cv2.imread(x,cv2.IMREAD_GRAYSCALE)
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
        #store raw image and mask
        self.storeRawImage(image)
        self.createAndStoreMask(image)
        #pour packLine in data buffer
        globalVariables.dataBuffer[self.bufferPointer% globalVariables.bufferSize:(self.bufferPointer% globalVariables.bufferSize)+globalVariables.linePackSize//2,
                                    globalVariables.borderSize:globalVariables.sensorSize+globalVariables.borderSize] = image[:globalVariables.linePackSize//2, :]
        self.bufferPointer = (self.bufferPointer + globalVariables.linePackSize//2)
        globalVariables.dataBuffer[self.bufferPointer% globalVariables.bufferSize:(self.bufferPointer% globalVariables.bufferSize)+globalVariables.linePackSize//2,
                                    globalVariables.borderSize:globalVariables.sensorSize+globalVariables.borderSize] = image[globalVariables.linePackSize//2:, :]
        self.bufferPointer = (self.bufferPointer + globalVariables.linePackSize//2)


    def dataBufferToProcessBuffer(self):
        
        #process Buffer is empty
        #compute start and end of super patch in data buffer
        s = self.startOfBuffer % globalVariables.bufferSize
        end = globalVariables.superPatchSize - (globalVariables.bufferSize - s)
        if s == 0:
            globalVariables.processBuffer = globalVariables.dataBuffer[s:globalVariables.superPatchSize, :]
        else:
            globalVariables.processBuffer = np.concatenate((globalVariables.dataBuffer[s:, :],
                                                           globalVariables.dataBuffer[:end, :]), axis=0)
        
        #fill triangles resulting from rotation and padding around image with sheet border value
        cv2.imwrite(globalVariables.outputPath+"/rawImage%d.png"%self.i,globalVariables.processBuffer)
        start_time = time.time()
        self.fillTriangles()
        print("fillTriangles:",time.time() - start_time)
        start_time = time.time()
        self.paddingLeftRightTopButtom()
        print("padding:",time.time() - start_time)
        cv2.imwrite(globalVariables.outputPath+"superP%d.png"%self.i,globalVariables.processBuffer)
        self.i +=1
        globalVariables.procBufferEmpty = 0 #set process Buffer is full
        globalVariables.patchCounter += 1
        self.startOfBuffer = self.startOfBuffer + globalVariables.superPatchSize
        globalVariables.procBufferEmpty = 1 #this is temporary
        globalVariables.processBuffer.fill(globalVariables.backgroundGrayLevel)


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
        #save all image and reset buffers
        globalVariables.patchCounter = 0
        globalVariables.rawImage.append(np.vstack(self.localRawImage))
        globalVariables.imageMask.append(np.vstack(self.localMask))
        globalVariables.cameraImage.append(np.vstack(self.localCameraImage))
        cv2.imwrite(globalVariables.outputPath+"backsuperP%d.png"%self.i,globalVariables.imageMask[0].astype(np.uint8))
        print("complete an image")
        globalVariables.imageMask.clear()
        self.localRawImage.clear()
        self.localMask.clear()
        self.localCameraImage.clear()
        globalVariables.endOfSheet = 1   

    def load_dataset_folder(self):
        x = []
        image_dir = globalVariables.inputPath
        files = os.listdir(image_dir)
        files.sort(key=lambda x: int(x.split('.')[0]))
        for image_type in files:
            # load images
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
        y, x, h, w = cv2.boundingRect(max_contour)
        x_min, x_max, y_min, y_max = x, x+w, y, y+h
                 
        # Find non-zero indices in rows and columns outside the loop
        nonzero_rows_indices = [np.nonzero(mask[i, :])[0] for i in range(x_min, x_max)]
        nonzero_cols_indices = [np.nonzero(mask[:, j])[0] for j in range(y_min, y_max)]

        for i in range(x_min, x_max):
            if np.any(mask[i,:] == 0):
                for j in range(y_min, y_max):
                    if mask[i, j] == 0:
                        minsR, maxesR = nonzero_cols_indices[j - y_min][0], nonzero_cols_indices[j - y_min][-1]
                        minsC, maxesC = nonzero_rows_indices[i - x_min][0], nonzero_rows_indices[i - x_min][-1]

                        
                        #left border
                        if i < minsR:
                            if j < minsC:
                                if (minsR - i) < (minsC - j):
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[minsR, j]
                                else:
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[i, minsC]

                            if j > maxesC:
                                if (minsR - i) < (j - maxesC):
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[minsR, j]
                                else:
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[i, maxesC]
                        #right border
                        if i > maxesR:
                            if j < minsC:
                                if (i - maxesR) < (minsC - j):
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[maxesR, j]
                                else:
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[i, minsC]
                                            
                            if j > maxesC:
                                if (i - maxesR) <= (j - maxesC):
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[maxesR, j]
                                else:
                                        globalVariables.processBuffer[i, j] = globalVariables.processBuffer[i, maxesC]
                
    
    def paddingLeftRightTopButtom(self):
        R, C = globalVariables.processBuffer.shape
        _, mask = cv2.threshold(globalVariables.processBuffer, self.minBackgroundLevel, globalVariables.backgroundGrayLevel, cv2.THRESH_BINARY_INV)
        kernel = np.ones((11,11), np.uint8)
        # Erode the binary image
        mask = cv2.erode(mask, kernel, iterations=1)

        #padding left and right
        # Find the minimum and maximum non-zero indices along each row
        min_indices = np.argmax(mask, axis=1)
        max_indices = C - np.argmax(np.flip(mask, axis=1), axis=1) - 1
        # Update processBuffer for padding left and right
        for r in range(R):
            min_val = globalVariables.processBuffer[r, min_indices[r]]
            max_val = globalVariables.processBuffer[r, max_indices[r]]
            globalVariables.processBuffer[r, :min_indices[r]] = min_val
            globalVariables.processBuffer[r, max_indices[r]+1:] = max_val
       
        #padding top and bottom
        _, mask = cv2.threshold(globalVariables.processBuffer, self.minBackgroundLevel, globalVariables.backgroundGrayLevel, cv2.THRESH_BINARY_INV)
        # Erode the binary image
        mask = cv2.erode(mask, kernel, iterations=1)
        ind = np.where(mask[:, 0] != 0)
        if len(ind[0]) > 0:
            minimum,maximum = min(ind[0]),max(ind[0])
            globalVariables.processBuffer[:minimum, :] = globalVariables.processBuffer[minimum, :]
            globalVariables.processBuffer[maximum:, :] = globalVariables.processBuffer[maximum, :] 



    def createAndStoreMask(self,image):
        (T, thresholded) = cv2.threshold(image, self.minBackgroundLevel, globalVariables.backgroundGrayLevel,cv2.THRESH_BINARY_INV )
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
        image = np.where(image > 220, 0, image)
        if  np.all(image == 0):
            return True
        else:
            return False
        
  
        



