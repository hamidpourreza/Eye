import globalVariables
from DataAcquisition import DataAcquisitionThread
from DataProcessing import DataProcessingThread
from ActionAndVisualization import ActionAndVisualizationThread
import numpy as np




# Create and start data acquisition thread
data_acquisition_thread = DataAcquisitionThread()
data_acquisition_thread.start()

# Create and start data processing thread
data_processing_thread = DataProcessingThread()
data_processing_thread.start()

# Create and start action and visualization thread
action_visualization_thread = ActionAndVisualizationThread()
action_visualization_thread.start()


