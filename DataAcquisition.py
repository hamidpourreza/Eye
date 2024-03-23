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
        #where super patch started
        self.startOfBuffer = 0    
        self.localCameraImage = []
        self.localMask = []
        self.localRawImage = []
        self.localImage = []

        globalVariables.paperDetect = 0
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
        globalVariables.dataBuffer[:globalVariables.borderSize,:] = 255
        self.bufferPointer = globalVariables.borderSize
        #super patch 0 start from index 0
        self.startOfBuffer = 0

        for x in data:
            #change read data rate
            if globalVariables.procBufferEmpty == 0:
              time.sleep(1/globalVariables.readImageRate)
            #get linepack
            image = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2GRAY)
            self.storeCameraImage(image)
            globalVariables.lineReceived += globalVariables.linePackSize
                  
            if self.allLinesAreEmpty(image):
                #sheet is not started
                if globalVariables.paperDetect != 1:
                    continue

                #sheet is finished & fill buffer with background pack
                else:
                    #data buffer is ready for processing
                    if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                        #process buffer is full
                        if globalVariables.procBufferEmpty == 0:
                            print("non real time")
                            globalVariables.nonRealtime = 1
                            continue
                        #process buffer is empty
                        else:
                            self.dataBufferToProcessBuffer()

                            if globalVariables.patchCounter >= globalVariables.maxSheetLength:
                                self.endOfSheet()
                                continue
                            elif np.all(image[-1, :] > 220):
                                #fill top border of first super patch
                                globalVariables.dataBuffer[:globalVariables.borderSize,:] = 255
                                self.bufferPointer = globalVariables.borderSize
                                #super patch 0 start from index 0
                                self.startOfBuffer = 0
                                globalVariables.paperDetect = 0
                                self.endOfSheet()
                                continue
                            else:
                                continue
                    #data buffer is full
                    else:
                        self.SavePackLine(image)
                        #data buffer is ready for processing
                        if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                            #process buffer is full
                            if globalVariables.procBufferEmpty == 0:
                                print("non real time")
                                globalVariables.nonRealtime = 1
                                continue
                            #process buffer is empty
                            else:
                                self.dataBufferToProcessBuffer()

                                if globalVariables.patchCounter >= globalVariables.maxSheetLength:
                                    self.endOfSheet()
                                    continue
                                elif np.all(image[-1, :] > 220):
                                    #fill top border of first super patch
                                    globalVariables.dataBuffer[:globalVariables.borderSize,:] = 255
                                    self.bufferPointer = globalVariables.borderSize
                                    #super patch 0 start from index 0
                                    self.startOfBuffer = 0
                                    globalVariables.paperDetect = 0

                                    self.endOfSheet()
                                    continue
                                else:
                                    continue
                            
                        else:
                            continue


                  
            #sheet is seen    
            else:
                globalVariables.paperDetect = 1

                #check data buffer has space and put the lines in it
                if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:
                        #data buffer is ready
                        if globalVariables.procBufferEmpty == 0:
                            #process buffer is full
                            print("non real time")
                            globalVariables.nonRealtime = 1
                            continue

                        else:
                            #put super patch in process buffer
                            
                            self.dataBufferToProcessBuffer()

                            if globalVariables.patchCounter >= globalVariables.maxSheetLength:
                                    self.endOfSheet()
                                    continue
                            elif np.all(image[-1, :] > 220):
                                    #fill top border of first super patch
                                    globalVariables.dataBuffer[:globalVariables.borderSize,:] = 255
                                    self.bufferPointer = globalVariables.borderSize
                                    #super patch 0 start from index 0
                                    self.startOfBuffer = 0
                                    globalVariables.paperDetect = 0

                                    self.endOfSheet()
                                    continue
                            else:
                                    continue
                                    
                #data buffer has space for pack
                else:
                    self.SavePackLine(image)
                    #check again data buffer is ready
                    if self.bufferPointer - self.startOfBuffer >= globalVariables.superPatchSize:

                        if globalVariables.procBufferEmpty == 0:
                                #process buffer is full
                                print("non real time")
                                globalVariables.nonRealtime = 1
                                continue
                        else:
                                #put super patch in process buffer
                            
                                self.dataBufferToProcessBuffer()

                                if globalVariables.patchCounter >= globalVariables.maxSheetLength:
                                        self.endOfSheet()
                                elif np.all(image[-1, :] > 220):
                                        #fill top border of first super patch
                                        globalVariables.dataBuffer[:globalVariables.borderSize,:] = 255
                                        self.bufferPointer = globalVariables.borderSize
                                        #super patch 0 start from index 0
                                        self.startOfBuffer = 0
                                        globalVariables.paperDetect = 0

                                        self.endOfSheet()
                                        continue
                                else:
                                        continue
                    else:
                        continue
 
                


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
        self.fillTriangles()
        self.paddingLeftRightTopButtom()
        cv2.imwrite(globalVariables.outputPath+"superP%d.png"%self.i,globalVariables.processBuffer)
        self.i +=1
        globalVariables.procBufferEmpty = 0 #set process Buffer is full
        globalVariables.patchCounter += 1
        self.startOfBuffer = self.startOfBuffer + globalVariables.superPatchSize
        globalVariables.procBufferEmpty = 1 #this is temporary



    def endOfSheet(self):
        #save all image and reset buffers
        globalVariables.patchCounter = 0
        globalVariables.rawImage.append(np.vstack(self.localRawImage))
        globalVariables.imageMask.append(np.vstack(self.localMask))
        globalVariables.cameraImage.append(np.vstack(self.localCameraImage))
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
        globalVariables.processBuffer = np.where(globalVariables.processBuffer > 220, 0, globalVariables.processBuffer)
        mask = np.where(globalVariables.processBuffer != 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)> 0:
            max_contour = max(contours, key=cv2.contourArea)
            # Find the bounding rectangle of the sheet
            y, x, h, w = cv2.boundingRect(max_contour)
            
            x_min, x_max, y_min, y_max = x, x+w, y, y+h
            image1 = globalVariables.processBuffer.copy()
            indexes = np.where(globalVariables.processBuffer[x_min:x_max,y_min:y_max] == 0)
            #fill left and right triangle
            for i, j in zip(indexes[0],indexes[1]):
                        i , j = i+x_min,j+y_min
                        row = image1[i,:]
                        col = image1[:, j]
                        indexesRow = np.where(row != 0)
                        indexesCol = np.where(col != 0)
                        minsC,maxesC = min(indexesRow[0]),max(indexesRow[0])
                        minsR,maxesR = min(indexesCol[0]),max(indexesCol[0])
                        
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
        #padding left and right
        for r in range(R):
          row = globalVariables.processBuffer[r,:]
          if np.all(row == 0):
             continue
          ind = np.where(row != 0)
          minimum,maximum = min(ind[0]),max(ind[0])
          globalVariables.processBuffer[r,:minimum] = row[minimum]
          globalVariables.processBuffer[r,maximum:] = row[maximum] 
       
        #padding top and bottom
        nonzero_rows = np.where(np.any(globalVariables.processBuffer != 0, axis=1))[0]

        first_nonzero_row = nonzero_rows[0]
        last_nonzero_row = nonzero_rows[-1]
        for r in range(R):
          row = globalVariables.processBuffer[r,:]
          if np.all(row == 0):
              if r < first_nonzero_row:
                  globalVariables.processBuffer[r, :] = globalVariables.processBuffer[first_nonzero_row, :]
              elif r >= last_nonzero_row:
                  globalVariables.processBuffer[r, :] = globalVariables.processBuffer[last_nonzero_row-1, :]

    def createAndStoreMask(self,image):
        mask = np.where(image > 220, 0, 1)
        self.localMask.append(mask)


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
        
  
        



