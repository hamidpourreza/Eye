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
        self.localCameraImage = []
        self.localMask = []
        self.localRawImage = []
        self.localImage = []

        globalVariables.buffer1Ready = 0
        globalVariables.buffer2Ready = 0
        globalVariables.paperDetect = 0
        globalVariables.cameraDetect = 0
        globalVariables.nonRealtime = 0
        globalVariables.patchCounter = 0
        globalVariables.endOfSheet = 0
        globalVariables.lineReceived = 0

    def run(self):
        i = 0
        data = self.load_dataset_folder()
        for x in data:
            if globalVariables.buffer1Ready == 1 or globalVariables.buffer2Ready == 1:
              time.sleep(1/globalVariables.readImageRate)
            #get linepack
            image = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2GRAY)
            
            self.storeCameraImage(image)
            globalVariables.lineReceived += globalVariables.linePackSize
                  
            if self.allLinesAreEmpty(image):
                #sheet is not started
                if globalVariables.paperDetect != 1:
                    continue

                #sheet is finished
                else:
                    
                    #padding end of 
                    self.fillCurrentBuffer()
                    self.bufferPointer = 0
                    globalVariables.paperDetect = 0
                    self.firstPatch = 1
                    globalVariables.patchCounter = 0
                    globalVariables.rawImage.append(np.vstack(self.localRawImage))
                    self.localRawImage.clear()
                    globalVariables.imageMask.append(np.vstack(self.localMask))
                    self.localMask.clear()
                    globalVariables.endOfSheet = 1

                    
            #sheet is seen    
            else:
                globalVariables.paperDetect = 1
                i = i +1
                #check which buffer has space and put the lines in it
                if 0 <= (self.bufferPointer- globalVariables.patchSize) <  globalVariables.linePackSize:
                    #No buffer has free space
                    if globalVariables.buffer2Ready == 1:
                        globalVariables.nonRealtime = 1
                        print("nonRealTime")
                        continue
                    #buffer2 has space
                    else:
                        self.createAndStoreMask(image)
                        self.storeRawImage(image)
                        image = self.fillTriangles(image)
                        image = self.paddingLeftRightTopButtom(image)
                        globalVariables.superPatchBuffer[self.bufferPointer:self.bufferPointer +  globalVariables.linePackSize,:] = image
                        self.bufferPointer = (self.bufferPointer +  globalVariables.linePackSize) % ( 2 * globalVariables.patchSize)
                
                elif 0 <= self.bufferPointer <  globalVariables.linePackSize:
                        #No buffer has free space
                        if globalVariables.buffer1Ready == 1:
                            globalVariables.nonRealtime = 1
                            print("nonRealTime")
                            continue
                        #buffer1 has space
                        else:
                            self.createAndStoreMask(image)
                            self.storeRawImage(image)
                            image = self.fillTriangles(image)
                            image = self.paddingLeftRightTopButtom(image)
                            globalVariables.superPatchBuffer[self.bufferPointer:self.bufferPointer +  globalVariables.linePackSize,:] = image
                            self.bufferPointer = (self.bufferPointer +  globalVariables.linePackSize) % (2* globalVariables.patchSize)
                
                #There are spaces
                else:
                        self.createAndStoreMask(image)
                        self.storeRawImage(image)
                        image = self.fillTriangles(image)
                        image = self.paddingLeftRightTopButtom(image)
                        globalVariables.superPatchBuffer[self.bufferPointer:self.bufferPointer +  globalVariables.linePackSize,:] = image
                        self.bufferPointer = (self.bufferPointer +  globalVariables.linePackSize) % (2* globalVariables.patchSize)
                
                #check buffer1 is ready
                if 0 <= (self.bufferPointer-( globalVariables.patchSize+ globalVariables.borderSize)) <  globalVariables.linePackSize:
                    #buffer1 is ready
                    if self.firstPatch == 1:
                        #padding beginning of buffer 1
                        globalVariables.superPatchBuffer[2*globalVariables.patchSize - globalVariables.borderSize: , :] = globalVariables.superPatchBuffer[0,:]
                        self.firstPatch = 0
                    globalVariables.buffer1Ready = 1
                    globalVariables.patchCounter +=1
                
                    
                    if globalVariables.patchCounter != globalVariables.maxSheetLength:
                        continue
                    else:
                        self.bufferPointer = 0
                        globalVariables.patchCounter = 0
                        globalVariables.rawImage.append(np.vstack(self.localRawImage))
                        self.localRawImage.clear()
                        globalVariables.imageMask.append(np.vstack(self.localMask))
                        self.localMask.clear()
                        globalVariables.endOfSheet = 1

                #chech buffer2 is ready
                elif 0 <= (self.bufferPointer- globalVariables.borderSize) <  globalVariables.linePackSize:
                        #buffer2 is not ready
                        if self.firstPatch == 1:
                            continue
                        #buffer2 is ready
                        else:
                            globalVariables.buffer2Ready = 1
                            globalVariables.patchCounter +=1
                    
                            if globalVariables.patchCounter != globalVariables.maxSheetLength:
                                continue
                            else:
                                self.bufferPointer = 0
                                globalVariables.patchCounter = 0
                                globalVariables.rawImage.append(np.vstack(self.localRawImage))
                                self.localRawImage.clear()
                                globalVariables.imageMask.append(np.vstack(self.localMask))
                                self.localMask.clear()
                                globalVariables.endOfSheet = 1
          
                
                
                

                            


    # Functions     
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

    def fillTriangles(self,image):
        image = np.where(image > 220, 0, image)
        mask = np.where(image != 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)> 0:
            max_contour = None
            max_area = 0
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)
                # Update if the current contour has a larger area
                if area > max_area:
                    max_area = area
                    max_contour = contour
            # Find the bounding rectangle of the sheet
            x, y, w, h = cv2.boundingRect(max_contour)
            
            x_min, x_max, y_min, y_max = x, x+w, y, y+h
            #fill left and right triangle
            for i in range(y_min, y_max):
                row = image[i,:]
                indexes = np.where(row != 0)
                mins,maxes = min(indexes[0]),max(indexes[0])
                if maxes-mins > (x_max-x_min)-50:
                    image[i,x_min:mins] = image[i,mins]
                    image[i,maxes:x_max] = image[i,maxes]
            #fill top and bottom triangle
            for i in range(x_min, x_max):
                col = image[:,i]
                indexes = np.where(col != 0)
                mins,maxes = min(indexes[0]),max(indexes[0])
                image[y_min:mins,i] = image[mins,i]
                image[maxes:y_max,i] = image[maxes,i]

        self.localImage.append(image)
            
        return image

    def paddingLeftRightTopButtom(self,image):
        R, C = image.shape
        #padding left and right
        for r in range(R):
          row = image[r,:]
          if np.all(row == 0):
             continue
          ind = np.where(row != 0)
          minimum,maximum = min(ind[0]),max(ind[0])
          image[r,:minimum] = row[minimum]
          image[r,maximum:] = row[maximum] 
        # Add borders
        image = np.concatenate((image[:, : globalVariables.borderSize], image),axis=1)
        image = np.concatenate((image,image[:, - globalVariables.borderSize:]),axis=1)
      
        #padding top and bottom
        nonzero_rows = np.where(np.any(image != 0, axis=1))[0]

        first_nonzero_row = nonzero_rows[0]
        last_nonzero_row = nonzero_rows[-1]
        for r in range(R):
          row = image[r,:]
          if np.all(row == 0):
              if r < first_nonzero_row:
                  image[r, :] = image[first_nonzero_row, :]
              elif r > last_nonzero_row:
                  image[r, :] = image[last_nonzero_row, :]
        return image

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
        
    def fillCurrentBuffer(self):
        if globalVariables.borderSize <= self.bufferPointer < globalVariables.patchSize:
            # Current buffer is buffer 1
            globalVariables.superPatchBuffer[self.bufferPointer:globalVariables.patchSize + globalVariables.borderSize, :] = globalVariables.superPatchBuffer[self.bufferPointer-1,:]
            globalVariables.buffer1Ready = 1

        elif globalVariables.patchSize <= self.bufferPointer < globalVariables.patchSize + globalVariables.borderSize:
            globalVariables.superPatchBuffer[self.bufferPointer:globalVariables.patchSize + globalVariables.borderSize, :] = globalVariables.superPatchBuffer[self.bufferPointer-1,:]
            globalVariables.buffer1Ready = 1

        elif globalVariables.patchSize + globalVariables.borderSize <= self.bufferPointer <  globalVariables.patchSize*2:
            # Current buffer is buffer 2
            globalVariables.superPatchBuffer[self.bufferPointer:, :] = globalVariables.superPatchBuffer[self.bufferPointer-1,:]
            globalVariables.superPatchBuffer[:globalVariables.borderSize, :] = globalVariables.superPatchBuffer[self.bufferPointer-1, :]
            globalVariables.buffer2Ready = 1

        elif 0 <= self.bufferPointer < globalVariables.borderSize:
             globalVariables.superPatchBuffer[self.bufferPointer:globalVariables.borderSize, :] = globalVariables.superPatchBuffer[self.bufferPointer-1,:]
             globalVariables.buffer2Ready = 1
  
        



