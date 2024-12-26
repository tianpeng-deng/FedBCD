from torch.utils.data import Dataset
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import glob


def z_score_norm(img):
    u = np.mean(img)
    s = np.std(img)
    img = img - u
    if s == 0:
        return img
    return img/s


def min_max_norm(img, epsilon=1e-5):
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv + epsilon) / (maxv - minv + epsilon)


def get_img_label_paths(data_file):
    img_label_plist = []
    with open(data_file, 'r') as f:
        for l in f:
            img_label_plist.append(l.strip().split(','))
    return img_label_plist[1:]

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)   # spacing格式转换
    resampler.SetReferenceImage(itkimage)   # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

def resize_3d(img, resize_shape, order=0):
    zoom0 = resize_shape[0] // img.shape[0]
    zoom1 = resize_shape[1] // img.shape[1]
    zoom2 = resize_shape[2] // img.shape[2]
    img = zoom(img, (zoom0, zoom1, zoom2), order=order)
    return img


def get_img(img_path):
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk)
    return img


def save_img(img, save_path):
    img_itk = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_itk, save_path)


class Dataset_Video(Dataset):
    def __init__(self, data_dir, data_file, input_shape, transforms=None, target_transforms=None):
        super(Dataset_Video, self).__init__()
        self.data_dir = data_dir
        self.img_label_plist = get_img_label_paths(data_file)
        self.input_shape = input_shape
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.label_dict = {'B': 0, 'M': 1}
        
    def __getitem__(self, index):
        _, y_cls, x_path, _ = self.img_label_plist[index]
        y_cls = self.label_dict[y_cls]
        x_path = x_path.replace('\\', '/')
        img_x = sitk.ReadImage(os.path.join(self.data_dir, x_path))
        img_x = resize_image_itk(img_x, self.input_shape, resamplemethod=sitk.sitkLinear)
        
        img_x = sitk.GetArrayFromImage(img_x)
        img_x = z_score_norm(img_x)
        img_x = min_max_norm(img_x)
        img_x = np.expand_dims(img_x, 0)
        if self.transforms is not None:
            img_x = self.transforms(img_x)
        return img_x.astype('float32'), np.array(y_cls, dtype=float)
    
    def __len__(self):
        return len(self.img_label_plist)

