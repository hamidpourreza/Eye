import cv2
from cv2.typing import RotatedRect
import numpy as np
# import imutils
import torchvision
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
# from main1 import Main
from Dataset import MyDataset as mvtec
import threading
import torch.cuda
import globalVariables


import math

class LineScanCameraSimulator:
  """
  Simulates a line scan camera.

  Args:
    num_pixels: The number of pixels in the camera.
    height: The number of lines that the camera sends at a time.
    background_gray_level: The grayscale level of the background.
    background_noise_variance: The variance of the Gaussian noise of the background.
    max_rotation: The maximum unwanted rotation of the image.
    blur_filter_size: The size of the blur filter to simulate camera defocus.
    bias: Bias to ensure the distance between two sheets
  """

  def __init__(self, num_pixels, height, background_gray_level, background_noise_variance, max_rotation, blur_filter_size,bias):
    self.num_pixels = num_pixels
    self.height = height
    self.background_gray_level = background_gray_level
    self.background_noise_variance = background_noise_variance
    self.max_rotation = max_rotation
    self.blur_filter_size = blur_filter_size
    self.bias = bias

  def simulate(self,image):
    image1 = image.squeeze(0).squeeze(0)
    #rotate image
    rotation = np.random.uniform(-self.max_rotation, self.max_rotation)
    border = self.background_gray_level + np.random.normal(0, self.background_noise_variance)
    rotated_image,mask_out = self.rotate_img(image1,rotation)
  
    if rotated_image.shape[1] > self.num_pixels:
       print("image width is grater of camera width")
       image = cv2.resize(rotated_image, (self.num_pixels, rotated_image.shape[0]))
    p_img = []
    # Randomly round the number of blocks
    before_blks = np.round(50 * np.random.rand()) + self.bias
    after_blks = np.round(50 * np.random.rand()) + self.bias
    # Calculate new image dimensions
    rows , cols = rotated_image.shape
    img_rows = self.height * np.ceil((self.height * (before_blks + after_blks) + rows) / self.height)
    img_columns = self.num_pixels
    background = np.double(self.background_gray_level * np.ones((int(img_rows), int(img_columns))))
    noise = np.sqrt(self.background_noise_variance) * np.random.randn(int(img_rows), int(img_columns))
    image= np.minimum(background + noise , 255)
    image = np.uint8(image)
    # cv2.imwrite("a3.png",image)
    min = np.min(image)
    max = np.max(image)
    print(min,max)
    # Random deviation from center
    deviation_from_center = np.round((1 - 2 * np.random.rand()) * 50)

    # Calculate the column transformation
    c_trans = int(deviation_from_center + (self.num_pixels - cols) / 2)

    # Paste the rotated image onto the background
    for r in range(rows):
        for c in range(cols):
            if mask_out[r, c] == True:  # Assuming white pixels in the mask represent background
               image[int(self.height * before_blks) + r, c_trans + c] = rotated_image[r, c]


  
    # Apply the blur filter to simulate camera defocus.
    if self.blur_filter_size > 1:
      image = cv2.GaussianBlur(image, (self.blur_filter_size, self.blur_filter_size), 0)


    return image

  def rotate_img(self,img,rotation):
      R, C = img.shape
      center_in = (round(R/2), round(C/2))
      theta = -rotation * np.pi / 180  # deg -> rad
      rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

      rot_r = round(R * abs(np.cos(theta)) + C * abs(np.sin(theta)))
      rot_c = round(C * abs(np.cos(theta)) + R * abs(np.sin(theta)))
      center_out = (round(rot_r / 2), round(rot_c / 2))
      f_out = np.zeros((rot_r, rot_c), dtype=np.uint8)
      mask_out = np.zeros((rot_r, rot_c), dtype=bool)

      for r in range(rot_r):
          for c in range(rot_c):
              rd = r - center_out[0]
              cd = c - center_out[1]
              rcs = np.round([rd, cd] @ rot_mat)
              rs = int(rcs[0] + center_in[0])
              cs = int(rcs[1] + center_in[1])
              if 0 <= rs < R and 0 <= cs < C:
                  f_out[r, c] = img[rs, cs]
                  mask_out[r, c] = True

      f_out[~mask_out] = self.background_gray_level
      return  f_out,mask_out


# Example usage
root_path = "/content/drive/MyDrive/AnomalyDetection/data/"
output_path = globalVariables.inputPath
batch_size = 1
test_dataset = mvtec(root_path= root_path,is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

simulator = LineScanCameraSimulator(
  num_pixels=6*1024,
  height=32,
  background_gray_level=255,
  background_noise_variance=10,
  max_rotation=3,
  blur_filter_size=11,
  bias = 10
)
images = []
for x in tqdm(test_dataloader, 'camera'):
  print(x.shape)
  simulated_image = simulator.simulate(x)
  images.append(simulated_image)
images = np.vstack(images)
print(images.shape)
for i in range(0,images.shape[0],32):
  cv2.imwrite(output_path +"/%i.png"%(i//32) ,images[i:i+32,:])
cv2.imwrite("a2.png", images)

