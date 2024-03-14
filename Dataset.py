import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2


class MyDataset(Dataset):
    def __init__(self, root_path, is_train=True,resize=320, cropsize=320):
        self.root_path = root_path
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.folder_path = root_path

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        
        # set transforms
        self.transform_x = T.Compose([#T.Resize(resize, Image.ANTIALIAS),
                                      #T.CenterCrop(cropsize),
                                      #T.ToTensor(),
                                      T.Normalize(mean=[0.485],
                                                  std=[0.229])])
        self.transform_mask = T.Compose([#T.Resize(resize, Image.NEAREST),
                                         #T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x = self.x[idx]
        # x = Image.open(x).convert('L')
        x = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY)
        # print(x.shape)
        # x = self.transform_x(x)
        return x

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []
        img_dir = self.folder_path
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            x.append(img_type_dir)
        return list(x), list(y), list(mask)


