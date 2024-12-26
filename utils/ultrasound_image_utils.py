from torch.utils.data.dataset import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class Custom_Dataset(Dataset):
    def __init__(self, root, csv_path, transform, **kwargs):
        super().__init__()
        self.root = root
        self.label_bias = None
        if transform == None:
            raise ValueError('"tranform" is None, please use "utils.set_transform->transform"')
        self.transform = transform
        self.csv = csv_path
        df = pd.read_csv(self.csv, index_col=0)

        # ======= US images =========
        self.pids = df.iloc[:,0:1].pid.values
        self.file_name = df.iloc[:,2:3].US_file.values
        
        df['US_file'] = df['US_file'].map(lambda item: root+f'/{item}')
        self.x_data=df.iloc[:,2:3]
        self.y_data=df.iloc[:,1:2]
        
        self.info = df
        
        self.file_path = self.x_data.US_file.values
        self.y_label = self.y_data.appliance.values

    def __getitem__(self, idx):
        pids = self.pids[idx]
        names = self.file_name[idx]

        file_path = self.file_path[idx]
        label = self.y_label[idx]
        img = Image.open(file_path).convert('RGB')
        
        img = self.transform(img)

        return img.float(), np.array(label, dtype=float)

    def __len__(self):
        return len(self.info)


def inver_pixels(img):
    pixels = np.array(img)
    pixels = 1 - pixels
    return Image.fromarray(pixels)

def set_transform(img_size, transform_list):
    if 'centerCrop' in transform_list:
        tf_list = [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
    else:
        if isinstance(img_size, int) == 1:
            tf_list = [transforms.Resize((img_size, img_size))]
        elif len(img_size) == 2:
            tf_list = [transforms.Resize(img_size)]
    if 'gray' in transform_list:
        tf_list.append(transforms.Grayscale(num_output_channels=1))
    if 'invert' in transform_list:
        tf_list.append(transforms.Lambda(lambda x: inver_pixels(x)))
    if 'randomHflip' in transform_list:
        tf_list.append(transforms.RandomHorizontalFlip())
    if 'randomVflip' in transform_list:
        tf_list.append(transforms.RandomVerticalFlip())
    if 'colorjitter' in transform_list:
        tf_list.append(transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)))
    if 'totensor' in transform_list:
        tf_list.append(transforms.ToTensor())
    
    # print("Transform List: ", tf_list)
    return transforms.Compose(tf_list)