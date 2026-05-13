# -*- coding: utf-8 -*-
# @Time    : 2025/06/27
# @Author  : Grok (基于 Geng Qin 的代码修改)
# @File    : PLGC_dataset.py
# @Description: 胃癌高光谱数据的三分类数据集加载器（优化版，减少 RAM 占用）

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler

# 定义全局设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def open_file(datapath):
    """
    直接加载 .npy 文件到内存，优化为最小 RAM 占用。
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"文件 {datapath} 不存在。")

    try:
        data = np.load(datapath, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"加载文件 {datapath} 失败：{e}")

    if data.ndim != 3:
        raise ValueError(f"不支持的 .npy 文件形状：{data.shape}。预期是 3D 数组。")

    H, W, C = data.shape if data.shape[2] in [40, 48] else (data.shape[1], data.shape[2], data.shape[0])

    if C not in [40, 48]:
        raise ValueError(f"不支持的 .npy 文件通道数：{C}。预期是 40 或 48。")

    if data.shape[2] not in [40, 48]:
        data = np.transpose(data, (1, 2, 0))  # [C, H, W] -> [H, W, C]

    if C == 48:
        data = data[:, :, :40]
        print(f"警告：文件 {datapath} 通道数为 48，已截取前 40 个通道。")

    return data

# GPU 上的数据增强函数
def gpu_light_aug(data, seed=42):
    """
    轻度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
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
    return data

class PLGC_HSIClassification(Dataset):
    def __init__(self, root, txt_name: str = "PLGC_flod82train.txt", training=True, transforms=None):
        super(PLGC_HSIClassification, self).__init__()
        self.root = root
        self.training = training
        self.transforms = transforms

        self.label_value = {"NORMAL": 0, "GIN": 1, "IM": 2}
        txt_path = os.path.join("/home/workingkarl/work/code/SSTM/dataset", txt_name)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file {txt_path} does not exist")

        self.images = []
        self.labels = []
        self.problem_files = []

        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('/')
                if len(parts) != 2:
                    print(f"Skipping invalid line in {txt_name}: {line}")
                    continue
                folder, filename = parts
                folder = folder.upper()
                if folder not in self.label_value:
                    print(f"Skipping invalid folder in {txt_name}: {folder}")
                    continue
                label = self.label_value[folder]
                image_path = os.path.join(self.root, line + '.npy')
                if not os.path.exists(image_path):
                    print(f"Skipping missing file: {image_path}")
                    continue
                self.images.append(image_path)
                self.labels.append(label)

        if len(self.images) == 0:
            raise ValueError(f"No valid data found in {txt_path}")
        assert len(self.images) == len(self.labels), "Images and labels count mismatch"
        print(f"Loaded {len(self.images)} samples from {txt_name}, labels distribution: {[self.labels.count(i) for i in range(3)]}")

        # 设置 transforms 为 GPU 函数
        if self.transforms == 'light_aug':
            self.transforms = gpu_light_aug
        elif self.transforms == 'medium_aug':
            self.transforms = gpu_medium_aug
        elif self.transforms == 'high_aug' or self.transforms == 'strong_aug':
            self.transforms = gpu_high_aug

    def __getitem__(self, index):
        # 按需加载数据
        try:
            img = open_file(self.images[index])
        except Exception as e:
            print(f"Error loading {self.images[index]}: {str(e)}")
            self.problem_files.append(self.images[index])
            img = np.zeros((256, 256, 40), dtype=np.float32)

        label = self.labels[index]

        # 转换为 tensor 并立即移至 GPU
        data = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).to(device)

        # 在 GPU 上进行归一化
        data_min = data.min()
        data -= data_min
        data_max = data.max()
        if data_max > 0:
            data /= (data_max + 1e-6)

        # 在 GPU 上应用数据增强
        if self.training and self.transforms:
            data = self.transforms(data, seed=index)

        return data, label

    def __len__(self):
        return len(self.images)

def get_train_loader(dataset, batch_size, num_workers, sampler=None, collate_fn=None):
    """
    创建优化后的数据加载器。
    """
    is_shuffle = True if dataset.training else False
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # 限制为 0，避免多进程 RAM 占用
        drop_last=True,
        shuffle=is_shuffle if sampler is None else False,
        pin_memory=False,  # 禁用 pin_memory，减少 RAM 占用
        sampler=sampler,
        collate_fn=collate_fn
    )
    return train_loader

if __name__ == "__main__":
    root_path = "/home/workingkarl/work/dataset/PLGC_256"
    txt_dir = "/home/workingkarl/work/code/SSTM/dataset"

    train_dataset = PLGC_HSIClassification(
        root=root_path,
        txt_name='PLGC_flod82train.txt',
        transforms='high_aug',
        training=True
    )

    test_dataset = PLGC_HSIClassification(
        root=root_path,
        txt_name='PLGC_flod82test.txt',
        transforms=None,
        training=False
    )

    sample_weights = [1.0 / (train_dataset.labels.count(i) + 1e-6) for i in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = get_train_loader(train_dataset, batch_size=2, num_workers=0, sampler=sampler)
    test_loader = get_train_loader(test_dataset, batch_size=2, num_workers=0)

    print(f"训练集大小：{len(train_dataset)}")
    print(f"测试集大小：{len(test_dataset)}")
    for data, label in train_loader:
        print("训练批次形状：", data.shape, label.shape)
        break
    for data, label in test_loader:
        print("测试批次形状：", data.shape, label.shape)
        break