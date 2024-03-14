import threading
import globalVariables
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class ActionAndVisualizationThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.numberSheet = 0
    

    def run(self):
        while True:
            # Wait for data to be processed and ready for action or visualization
            if globalVariables.resultReady == 1:
                #caculate enhanced Image....

                scoreMap = globalVariables.scoreMap.pop(0)
                imageMask = globalVariables.imageMask.pop(0)
                rawImage = globalVariables.rawImage.pop(0)
                scoreMap = scoreMap[:imageMask.shape[0],:]
               
                cv2.imwrite(globalVariables.outputPath+"imageMask%d.png"%self.numberSheet,(imageMask*255).astype(np.uint8))
                cv2.imwrite(globalVariables.outputPath+"rawImage%d.png"%self.numberSheet,rawImage)

                #applay imageMask to scoreMap
                
                scoreMap0 = self.normalization(scoreMap)
                cv2.imwrite(globalVariables.outputPath+"scoreMap%d.png"%self.numberSheet,scoreMap0.astype(np.uint8))
                result = (scoreMap * imageMask)
                #Calculate defectMap
                defectMap = np.where(result > globalVariables.threshold, 255, 0)
                cv2.imwrite(globalVariables.outputPath+"defectMap%d.png"%self.numberSheet,defectMap.astype(np.uint8))
                result = self.normalization(result)
                cv2.imwrite(globalVariables.outputPath+"result%d.png"%self.numberSheet,result.astype(np.uint8))

                #Overlay defectMap on rawImage
                overlay = rawImage.copy()
                overlay[defectMap != 255] = 0
                cv2.imwrite(globalVariables.outputPath+"overlay%d.png"%self.numberSheet,overlay.astype(np.uint8))
                self.visualize_loc_result([rawImage],[result], globalVariables.threshold,globalVariables.outputPath)
                self.numberSheet += 1
                globalVariables.resultReady = 0
                print("finish")


    def visualize_loc_result(self,test_imgs, score_map_list, threshold,
                            save_path, vis_num=1):
        for t_idx in range(vis_num):
            test_img = test_imgs[t_idx]
            heat = score_map_list[t_idx]
            non_zero_row = ~np.all(heat == 0 , axis=1)
            non_zero_col = ~np.all(heat == 0 , axis=0)
            heat = heat[non_zero_row,:]
            heat = heat[:,non_zero_col]
            test_img = test_img[non_zero_row,:]
            test_img = test_img[:,non_zero_col]
            cv2.imwrite("1.png",test_img)
            test_pred = score_map_list[t_idx]
            test_pred = test_pred[non_zero_row,:]
            test_pred = test_pred[:,non_zero_col]
            test_pred[test_pred <= threshold] = 0
            test_pred[test_pred > threshold] = 1
            test_pred_img = test_img.copy()
            test_pred_img[test_pred == 0] = 0

            fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 4))
            fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

            ax_img[0].imshow(test_img)
            ax_img[0].set_title('Image')
            ax_img[1].imshow(heat)
            ax_img[1].set_title('HeatMap')
            ax_img[2].imshow(test_pred_img)
            ax_img[2].set_title('Predicted anomalous image')
            
            os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
            fig_img.savefig(os.path.join(save_path, 'images', '%03d.png' % ( t_idx)))
            fig_img.clf()
            plt.close(fig_img)

    def normalization(self,image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = ((image - min_val)/(max_val - min_val))*255
        return normalized_image




   