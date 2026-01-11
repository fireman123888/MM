"""
Caltech101-7 数据集加载器
支持半监督学习的多视图数据加载
"""
from typing import Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from data_loader import MultiViewDataset, SemiSupervisedDataLoader


class Caltech101Dataset:
    """Caltech101-7 数据集"""

    def __init__(self, data_path: str = 'Caltech101-7.mat', random_seed: int = 42):
        """
        Args:
            data_path: .mat 文件路径
            random_seed: 随机种子
        """
        self.data_path = data_path
        self.random_seed = random_seed

        # 加载数据
        print(f"正在加载 {data_path}...")
        mat_data = sio.loadmat(data_path)

        # 提取特征和标签
        self.X = mat_data['X']  # (6, 1) object array
        self.Y = mat_data['Y'].flatten()  # (1474,)

        # 提取所有视图
        self.num_views = self.X.shape[0]
        self.views_data = []
        for i in range(self.num_views):
            view = self.X[i][0].astype(np.float32)
            self.views_data.append(view)

        self.num_samples = self.views_data[0].shape[0]
        self.num_classes = len(np.unique(self.Y))
        self.view_dims = [view.shape[1] for view in self.views_data]

        print(f"数据集加载完成:")
        print(f"  样本数量: {self.num_samples}")
        print(f"  类别数量: {self.num_classes}")
        print(f"  视图数量: {self.num_views}")
        print(f"  视图维度: {self.view_dims}")

        # 标签转换为 0-based
        self.Y = self.Y - 1

    def load_data(self, labeled_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        划分数据集为有标签、无标签和测试集

        Args:
            labeled_ratio: 有标签数据比例
            test_ratio: 测试集比例

        Returns:
            labeled_dataset, unlabeled_dataset, test_dataset
        """
        np.random.seed(self.random_seed)

        # 随机打乱索引
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)

        # 划分数据集
        test_size = int(self.num_samples * test_ratio)
        train_size = self.num_samples - test_size

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # 从训练集中划分有标签和无标签
        num_labeled = int(train_size * labeled_ratio)
        labeled_indices = train_indices[:num_labeled]
        unlabeled_indices = train_indices[num_labeled:]

        print(f"\n数据集划分:")
        print(f"  有标签数据: {len(labeled_indices)} ({labeled_ratio*100:.1f}% of train)")
        print(f"  无标签数据: {len(unlabeled_indices)}")
        print(f"  测试数据: {len(test_indices)}")
        print(f"  总训练数据: {len(train_indices)}")

        # 创建数据集
        labeled_views = [view[labeled_indices] for view in self.views_data]
        labeled_labels = self.Y[labeled_indices]
        labeled_dataset = MultiViewDataset(labeled_views, labeled_labels, is_labeled=True)

        unlabeled_views = [view[unlabeled_indices] for view in self.views_data]
        unlabeled_labels = self.Y[unlabeled_indices]
        unlabeled_dataset = MultiViewDataset(unlabeled_views, unlabeled_labels, is_labeled=False)

        test_views = [view[test_indices] for view in self.views_data]
        test_labels = self.Y[test_indices]
        test_dataset = MultiViewDataset(test_views, test_labels, is_labeled=True)

        # 打印标签分布
        print(f"\n标签分布:")
        print(f"  有标签数据: {np.bincount(labeled_labels)}")
        print(f"  无标签数据: {np.bincount(unlabeled_labels)}")
        print(f"  测试数据: {np.bincount(test_labels)}")

        return labeled_dataset, unlabeled_dataset, test_dataset


def create_caltech_dataloaders(
    data_path: str = 'Caltech101-7.mat',
    labeled_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    mu: int = 7,
    random_seed: int = 42
) -> Tuple[SemiSupervisedDataLoader, DataLoader, Caltech101Dataset]:
    """
    创建 Caltech101-7 数据集的数据加载器

    Args:
        data_path: .mat 文件路径
        labeled_ratio: 有标签数据比例
        test_ratio: 测试集比例
        batch_size: 批次大小
        mu: 无标签/有标签比例
        random_seed: 随机种子

    Returns:
        train_loader: 半监督训练数据加载器
        test_loader: 测试数据加载器
        dataset: Caltech101Dataset 对象
    """
    # 加载数据集
    dataset = Caltech101Dataset(data_path, random_seed)

    # 划分数据
    labeled_dataset, unlabeled_dataset, test_dataset = dataset.load_data(
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio
    )

    # 创建数据加载器
    train_loader = SemiSupervisedDataLoader(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        batch_size=batch_size,
        mu=mu,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, dataset


if __name__ == '__main__':
    # 测试数据加载
    train_loader, test_loader, dataset = create_caltech_dataloaders(
        labeled_ratio=0.1,
        batch_size=16
    )

    print(f"\n数据加载器测试:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    # 测试一个批次
    batch = next(iter(train_loader))
    print(f"\n批次数据:")
    print(f"  有标签样本数: {len(batch['labeled']['label'])}")
    print(f"  无标签样本数: {len(batch['unlabeled']['label'])}")
    print(f"  视图数量: {len(batch['labeled']['views'])}")
    for i, view in enumerate(batch['labeled']['views']):
        print(f"    视图 {i+1} shape: {view.shape}")
