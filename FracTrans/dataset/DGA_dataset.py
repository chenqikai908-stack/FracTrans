# -*- coding: utf-8 -*-
# @Time    : 2025/06/28
# @Author  : Grok
# @File    : DGA_dataset_cls.py
# @Description: 数据集加载器，用于高光谱数据（.mat文件）的二分类，从txt文件读取路径

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils import data
from scipy.io import loadmat
import gc

# 定义全局设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def open_file(datapath):
    """
    加载 .mat 文件，优化内存使用。
    - 直接转换为 float32 减少内存占用。
    """
    _, ext = os.path.splitext(datapath)
    ext = ext.lower()
    if ext == '.mat':
        data = loadmat(datapath)
        if 'data' in data:
            data = data['data'].astype(np.float32)  # 优化：立即转换为 float32
        elif 'image' in data:
            data = data['image'].astype(np.float32)
        else:
            raise ValueError(f"未在 .mat 文件中找到有效数据键: {datapath}")
        if data.ndim == 3 and data.shape[2] == 60:
            return data  # [H, W, 60]
        elif data.ndim == 3 and data.shape[0] == 60:
            data = np.transpose(data, (1, 2, 0))  # [60, H, W] -> [H, W, 60]
            return data
        else:
            raise ValueError(f"不支持的 .mat 文件形状: {data.shape}")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def gpu_light_aug(data, seed=42):
    """
    轻度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    noise = torch.normal(0, 0.01, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
    return data

def gpu_medium_aug(data, seed=42):
    """
    中度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    noise = torch.normal(0, 0.02, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
    return data

def gpu_high_aug(data, seed=42):
    """
    高度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[2])  # 水平翻转
    noise = torch.normal(0, 0.01, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
    return data

class HSI_Dataset(Dataset):
    def __init__(self, root, txt_path, training=True, transforms=None):
        super(HSI_Dataset, self).__init__()
        self.root = root
        self.txt_path = txt_path
        assert os.path.exists(self.txt_path), f"文件 '{self.txt_path}' 不存在。"

        self.images = []
        self.labels = []

        # 只读取路径和标签，不加载数据
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split('/')
                    if len(parts) == 2:
                        folder, filename = parts
                        label = 0 if folder == 'N_256' else 1 if folder == 'P_256' else -1
                        if label >= 0:
                            image_path = os.path.join(self.root, folder, f"{filename}.mat")
                            if os.path.exists(image_path):
                                self.images.append(image_path)
                                self.labels.append(label)
                            else:
                                print(f"警告: 文件不存在，跳过: {image_path}")
                        else:
                            print(f"警告: 文件夹名称无效，跳过: {line}")

        if len(self.images) == 0:
            raise ValueError("未找到有效数据，请检查txt文件和图像路径。")

        assert len(self.images) == len(self.labels), "图像和标签数量不匹配"
        self.is_training = training
        self.transforms = transforms

        # 映射增强函数到 GPU
        if self.transforms == 'light_aug':
            self.transforms = gpu_light_aug
        elif self.transforms == 'medium_aug':
            self.transforms = gpu_medium_aug
        elif self.transforms == 'high_aug' or self.transforms == 'strong_aug':
            self.transforms = gpu_high_aug

        # 打印类别分布
        class_counts = [self.labels.count(0), self.labels.count(1)]
        print(f"数据集加载完成: {len(self.images)} 张图像，通道数: 60，类别分布: {class_counts}")

    def __getitem__(self, index):
        # 按需加载单个图像
        try:
            img = open_file(self.images[index])
        except Exception as e:
            print(f"Error loading {self.images[index]}: {str(e)}")
            img = np.zeros((256, 256, 60), dtype=np.float32)

        label = self.labels[index]

        # 转换为 tensor 并立即移至 GPU
        data = torch.from_numpy(img).permute(2, 0, 1).to(device=device, dtype=torch.float32)

        # 在 GPU 上进行归一化
        data_min = data.min()
        data -= data_min
        data_max = data.max()
        if data_max > 0:
            data /= (data_max + 1e-6)

        # 在 GPU 上应用数据增强
        if self.is_training and self.transforms:
            data = self.transforms(data, seed=index)

        # 优化：释放 CPU 内存
        del img
        gc.collect()

        return data, label

    def __len__(self):
        return len(self.images)

def get_train_loader(dataset, batch_size, num_workers, sampler=None, collate_fn=None):
    is_shuffle = True if dataset.is_training else False
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=is_shuffle if sampler is None else False,
        pin_memory=False,
        sampler=sampler,
        collate_fn=collate_fn
    )
    return train_loader

if __name__ == "__main__":
    root_path = "/home/workingkarl/work/shixiong/shixiongcode/dataset(danguanai)/DGA_fenlei"
    txt_dir = "/home/workingkarl/work/myself_code/dataset"

    train_dataset = HSI_Dataset(
        root=root_path,
        txt_path=os.path.join(txt_dir, 'fold82train.txt'),
        transforms='high_aug',
        training=True
    )

    test_dataset = HSI_Dataset(
        root=root_path,
        txt_path=os.path.join(txt_dir, 'fold82test.txt'),
        transforms=None,
        training=False
    )

    sample_weights = [1.0 / (train_dataset.labels.count(i) + 1e-6) for i in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = get_train_loader(train_dataset, batch_size=2, num_workers=0, sampler=sampler)
    test_loader = get_train_loader(test_dataset, batch_size=2, num_workers=0)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    for data, label in train_loader:
        print("训练批次形状:", data.shape, label.shape)
        break
    for data, label in test_loader:
        print("测试批次形状:", data.shape, label.shape)
        break