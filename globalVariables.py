import numpy as np

#Path
inputPath = "./AnomalyDetection/data1"
outputPath = "./AnomalyDetection/output/"

# Constants
borderSize = 48
patchSize = 512
sensorSize = 6144
linePackSize = 32
maxSheetLength = 10
cameraImageLength = 20*1024
threshold = 15.5 #for scoreMap
readImageRate = 1/20 #read 1 image each 20 second

# Global Variables
buffer1Ready = 0
buffer2Ready = 0
paperDetect = 0
cameraDetect = 0
linePerSecond = 0
nonRealtime = 0
resultReady = 0
#superPatch1 : superPatchBuffer[2*patchSize - borderSize: , :] , superPatchBuffer[:patchSize+borderSize,:]
#superPatch2 : superPatchBuffer[patchSize - borderSize: , :] , superPatchBuffer[:borderSize,:]
superPatchBuffer = np.zeros((2*patchSize, 2*borderSize + sensorSize))
imageMask = []
rawImage = []
enhancedImage = []
cameraImage = []
scoreMap = []
patchCounter = 0
endOfSheet = 0
lineReceived = 0