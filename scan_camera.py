import cv2
import numpy as np
from tqdm import tqdm
import globalVariables
import os


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
    def __init__(self, num_pixels, height, background_gray_level, background_noise_variance, max_rotation, blur_filter_size, bias):
        self.num_pixels = num_pixels
        self.height = height
        self.background_gray_level = background_gray_level
        self.background_noise_variance = background_noise_variance
        self.max_rotation = max_rotation
        self.blur_filter_size = blur_filter_size
        self.bias = bias

    def simulate(self, image):
        #defocuse image
        if self.blur_filter_size > 1:
            image = cv2.GaussianBlur(image.astype(np.uint8), (self.blur_filter_size, self.blur_filter_size), 0)
     
        #rotate image
        rotated_image, mask_out = self.rotate_img(image)

        if rotated_image.shape[1] > self.num_pixels:
            print("Image width is greater than camera width")
            image = cv2.resize(rotated_image, (self.num_pixels, rotated_image.shape[0]))

        before_blks = np.round(50 * np.random.rand()) + self.bias
        after_blks = np.round(50 * np.random.rand()) + self.bias

        #add background to image
        rows, cols = rotated_image.shape
        img_rows = self.height * np.ceil((self.height * (before_blks + after_blks) + rows) / self.height)
        img_columns = self.num_pixels
        background = np.double(self.background_gray_level * np.ones((int(img_rows), int(img_columns))))
        noise = np.sqrt(self.background_noise_variance) * np.random.randn(int(img_rows), int(img_columns))
        image = np.minimum(background + noise, 255)

        deviation_from_center = np.round((1 - 2 * np.random.rand()) * 50)
        c_trans = int(deviation_from_center + (self.num_pixels - cols) / 2)

        for r in range(rows):
            image[int(self.height * before_blks) + r, c_trans:c_trans + cols] = np.where(mask_out[r], rotated_image[r], image[int(self.height * before_blks) + r, c_trans:c_trans + cols])

        
        return image.astype(np.uint8)

    def rotate_img(self, img):
        rotation = np.random.uniform(-self.max_rotation, self.max_rotation)
        rows, cols = img.shape
        center = (cols / 2, rows / 2)
        angle = -rotation  # Rotation angle (counter-clockwise)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Calculate the bounding box of the rotated image
        cos_theta = np.abs(rotation_matrix[0, 0])
        sin_theta = np.abs(rotation_matrix[0, 1])
        new_width = int((rows * sin_theta) + (cols * cos_theta))
        new_height = int((rows * cos_theta) + (cols * sin_theta))
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        rotated_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=self.background_gray_level)

        # Create mask
        mask_out = rotated_image != self.background_gray_level

        return rotated_image, mask_out

  
    def load_dataset_folder(self):
        x = []
        image_dir = globalVariables.rootPath
        files = os.listdir(image_dir)
        # files.sort(key=lambda x: int(x.split('.')[0]))
        for image_type in files:
            # load images
            image_type_dir = os.path.join(image_dir, image_type)
            x.append(image_type_dir)
        return list(x)


# Example usage
root_path = globalVariables.rootPath
output_path = globalVariables.inputPath

simulator = LineScanCameraSimulator(
  num_pixels=globalVariables.sensorSize,
  height=globalVariables.linePackSize,
  background_gray_level=globalVariables.backgroundGrayLevel,
  background_noise_variance=10,
  max_rotation=3,
  blur_filter_size=11,
  bias = 10
)
data = simulator.load_dataset_folder()
images = []
for x in tqdm(data, 'camera'):
  image = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2GRAY)
  simulated_image = simulator.simulate(image)
  images.append(simulated_image)

images = np.vstack(images)
for i in range(0,images.shape[0],32):
  cv2.imwrite(output_path +"/%i.png"%(i//32) ,images[i:i+32,:])

