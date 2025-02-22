import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from utils.utils import cvtColor, preprocess_input

import numpy as np
import torch
import torchvision.transforms as transforms
import random
class DeeplabDatasetUnlabel(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDatasetUnlabel, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        # jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages_Unlabel"), name + ".jpg"))
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg    = self.get_random_data(jpg, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])

        return jpg

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)

        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        # iw, ih  = image.size
        h, w    = input_shape

        iw, ih  = image.size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', [w, h], (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        return new_image

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        # new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        # scale = self.rand(0.25, 2)
        # if new_ar < 1:
        #     nh = int(scale*h)
        #     nw = int(nh*new_ar)
        # else:
        #     nw = int(scale*w)
        #     nh = int(nw/new_ar)
        # image = image.resize((nw,nh), Image.BICUBIC)
        #
        # #------------------------------------------#
        # #   翻转图像
        # #------------------------------------------#
        # # flip = self.rand()<.5
        # # if flip:
        # #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # #------------------------------------------#
        # #   将图像多余的部分加上灰条
        # #------------------------------------------#
        # dx = int(self.rand(0, w-nw))
        # dy = int(self.rand(0, h-nh))
        # new_image = Image.new('RGB', (w,h), (128,128,128))
        # new_image.paste(image, (dx, dy))
        # image = new_image
        # image_data      = np.array(image, np.uint8)
        #
        # #------------------------------------------#
        # #   高斯模糊
        # #------------------------------------------#
        # # blur = self.rand() < 0.25
        # # if blur:
        # #     image_data = cv2.GaussianBlur(image_data, (5, 5), 0)
        #
        # #------------------------------------------#
        # #   旋转
        # #------------------------------------------#
        # # rotate = self.rand() < 0.25
        # # if rotate:
        # #     center      = (w // 2, h // 2)
        # #     rotation    = np.random.randint(-10, 11)
        # #     M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
        # #     image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
        #
        # #---------------------------------#
        # #   对图像进行色域变换
        # #   计算色域变换的参数
        # #---------------------------------#
        # r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # #---------------------------------#
        # #   将图像转到HSV上
        # #---------------------------------#
        # hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        # dtype           = image_data.dtype
        # #---------------------------------#
        # #   应用变换
        # #---------------------------------#
        # x       = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        #
        # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        # return image_data


# DataLoader中collate_fn使用
def deeplab_dataset_collate_unbel(batch):
    images      = []
    for img in batch:
        images.append(img)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    return images




def rand_bbox(size, lam):
    """Generate random bounding box

    Args:
        size: Image size of (512, 512)
        lam: Cutmix lambda parameter

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    W, H = size[0], size[1]

    # Calculate cut size based on lambda
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Generate random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Get bounding box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix_images(images1, images2, alpha=1.0):
    """
    Perform CutMix on two batches of images with shape [8,3,512,512]

    Args:
        images1: First batch of images, tensor of shape [8,3,512,512]
        images2: Second batch of images, tensor of shape [8,3,512,512]
        alpha: Beta distribution parameter

    Returns:
        mixed_images: CutMix result
        lam: The mixing ratio
    """
    # Input validation
    # assert images1.shape == images2.shape == (8, 3, 512, 512), "Images must be of shape [8,3,512,512]"

    # Generate mixing ratio from beta distribution
    lam = np.random.beta(alpha, alpha)

    # Get random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox((512, 512), lam)

    # Create copy of images1 as base
    mixed_images = images1.clone()

    # Replace the region in images1 with the region from images2
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images2[:, :, bbx1:bbx2, bby1:bby2]

    # Calculate actual lambda based on area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (512 * 512))

    return mixed_images, lam

def cutout_images(images1):
    return transforms.RandomErasing(p=1,
                             ratio=(1, 1),
                             scale=(0.01, 0.05),
                             # scale=(0.01, 0.01),
                             value=127)(images1)

def SA(images1, images2):
    methods=['cutmix_images', 'cutout_images']
    random_method = random.choice(methods)
    if random_method=='cutmix_images':
        mixed_images, lam = cutmix_images(images1, images2, alpha=1.0)
        return mixed_images
    else:
        return cutout_images(images1)


def WA(images, p=0.5, rotation_range=(-30, 30),
                               scale_range=(0.8, 1.2), translation_range=(-0.1, 0.1)):
    """
    对批量图像数据进行随机几何变换

    参数:
        images: shape为[batch_size, channels, height, width]的图像张量
        p: 每种变换的概率
        rotation_range: 旋转角度范围(度)
        scale_range: 缩放范围
        translation_range: 平移范围(相对于图像尺寸的比例)

    返回:
        变换后的图像张量
    """
    device = images.device
    batch_size, channels, height, width = images.shape

    # 创建变换矩阵
    transform_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    # 随机旋转
    if np.random.random() < p:
        angles = np.random.uniform(rotation_range[0], rotation_range[1], batch_size)
        angles_rad = np.deg2rad(angles)
        cos_angles = torch.cos(torch.tensor(angles_rad, device=device))
        sin_angles = torch.sin(torch.tensor(angles_rad, device=device))

        rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation_matrix[:, 0, 0] = cos_angles
        rotation_matrix[:, 0, 1] = -sin_angles
        rotation_matrix[:, 1, 0] = sin_angles
        rotation_matrix[:, 1, 1] = cos_angles

        transform_matrix = torch.bmm(transform_matrix, rotation_matrix)

    # 随机缩放
    if np.random.random() < p:
        scales = torch.tensor(np.random.uniform(scale_range[0], scale_range[1],
                                                (batch_size, 2)), device=device)
        scale_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        scale_matrix[:, 0, 0] = scales[:, 0]
        scale_matrix[:, 1, 1] = scales[:, 1]

        transform_matrix = torch.bmm(transform_matrix, scale_matrix)

    # 随机平移
    if np.random.random() < p:
        translations = torch.tensor(np.random.uniform(translation_range[0], translation_range[1],
                                                      (batch_size, 2)), device=device)
        translation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        translation_matrix[:, :2, 2] = translations * torch.tensor([width, height],
                                                                   device=device)

        transform_matrix = torch.bmm(transform_matrix, translation_matrix)

    # 随机水平翻转
    if np.random.random() < p:
        flip_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        flip_matrix[:, 0, 0] = -1
        flip_matrix[:, 0, 2] = width
        transform_matrix = torch.bmm(transform_matrix, flip_matrix)

    # 生成网格
    grid = F.affine_grid(transform_matrix[:, :2], images.size(), align_corners=False)

    # 应用变换
    transformed_images = F.grid_sample(images, grid, align_corners=False,
                                       mode='bilinear', padding_mode='zeros')

    return transformed_images