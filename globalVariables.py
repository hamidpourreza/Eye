import numpy as np
import os

#Path
inputPath = "./data1"       #path of line pack
outputPath = "./output/"        #path of output
rootPath = "./data/"        #path of main images(befor simulated)
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Constants
borderSize = 48
patchSize = 512
sensorSize = 6144
linePackSize = 32
maxSheetLength = 10
cameraImageLength = 20*1024
threshold = 15.5 #for scoreMap
readImageRate = 1/30 #read 1 image each 20 second
# bufferSize: > patchSize+2*boarderSize AND  n*linePackSize >=  patchSize+boarderSize
# superPatchSize= patchSize+2*boarderSize
n = 18
bufferSize = n * linePackSize + borderSize  #number of rows for data buffer
superPatchSize = patchSize + 2*borderSize

# Global Variables
procBufferEmpty = 1
paperDetect = 0
cameraDetect = 0
linePerSecond = 0
nonRealtime = 0
resultReady = 0
dataBuffer = np.ones((bufferSize, sensorSize + 2*borderSize)) * 255
processBuffer = np.ones((superPatchSize, sensorSize + 2*borderSize)) * 255
imageMask = []
rawImage = []
enhancedImage = []
cameraImage = []
scoreMap = []
patchCounter = 0
endOfSheet = 0
lineReceived = 0