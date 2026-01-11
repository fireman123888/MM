"""
多视图半监督数据集和数据加载器
支持同时提供有标签数据(X)和无标签数据(U)
支持多个数据集: Caltech101-20, Caltech-5V, COIL20, CUB, Out_Scene, WebKB
"""
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional
import random
import os


class MultiViewDataset(Dataset):
    """
    多视图数据集基类
    支持有标签和无标签数据
    """
    def __init__(self,
                 views_data: List[np.ndarray],
                 labels: Optional[np.ndarray] = None,
                 is_labeled: bool = True):
        """
        Args:
            views_data: 列表，包含V个视图的数据
                       每个元素为 (num_samples, feature_dim_v)
            labels: 标签数组 (num_samples,)，如果是无标签数据则为None
            is_labeled: 是否为有标签数据
        """
        self.views_data = [torch.FloatTensor(view) for view in views_data]
        self.num_views = len(views_data)
        self.num_samples = len(views_data[0])
        self.is_labeled = is_labeled

        # 验证所有视图的样本数相同
        assert all(len(view) == self.num_samples for view in views_data), \
            "All views must have the same number of samples"

        if is_labeled:
            assert labels is not None, "Labeled data must have labels"
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = torch.full((self.num_samples,), -1, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            views: 列表，包含V个视图的数据
            label: 标签（如果是无标签数据则为-1）
            is_labeled: 是否为有标签数据
        """
        views = [view[idx] for view in self.views_data]
        label = self.labels[idx]

        return {
            'views': views,
            'label': label,
            'is_labeled': self.is_labeled,
            'index': idx
        }


class HandwrittenNumeralsDataset:
    """
    Handwritten Numerals (HW) 数据集
    包含6个视图：fou (Fourier), fac (Profile), kar (Karhunen), pix (Pixel), zer (Zernike), mor (Morphological)
    """
    def __init__(self,
                 data_dir: str = './data/handwritten',
                 num_labeled: int = 100,
                 num_unlabeled: int = 1900,
                 random_seed: int = 42):
        """
        Args:
            data_dir: 数据目录
            num_labeled: 有标签样本数量
            num_unlabeled: 无标签样本数量
            random_seed: 随机种子
        """
        self.data_dir = data_dir
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.random_seed = random_seed

        # 视图名称
        self.view_names = ['fou', 'fac', 'kar', 'pix', 'zer', 'mor']
        self.num_views = len(self.view_names)

        # 视图维度（HW数据集的标准维度）
        self.view_dims = {
            'fou': 76,   # Fourier coefficients
            'fac': 216,  # Profile correlations
            'kar': 64,   # Karhunen-Love coefficients
            'pix': 240,  # Pixel averages
            'zer': 47,   # Zernike moments
            'mor': 6     # Morphological features
        }

        self.num_classes = 10  # 0-9 数字

    def load_data(self, use_mock_data: bool = False):
        """
        加载HW数据集

        Args:
            use_mock_data: 是否使用模拟数据（用于测试）

        Returns:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            test_dataset: 测试数据集
        """
        if use_mock_data:
            return self._generate_mock_data()
        else:
            return self._load_real_data()

    def _generate_mock_data(self):
        """生成模拟数据用于测试"""
        np.random.seed(self.random_seed)

        total_samples = self.num_labeled + self.num_unlabeled

        # 生成所有视图的数据
        all_views_data = []
        for view_name in self.view_names:
            view_dim = self.view_dims[view_name]
            view_data = np.random.randn(total_samples, view_dim).astype(np.float32)
            all_views_data.append(view_data)

        # 生成标签
        all_labels = np.random.randint(0, self.num_classes, size=total_samples)

        # 划分有标签和无标签数据
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        labeled_indices = indices[:self.num_labeled]
        unlabeled_indices = indices[self.num_labeled:self.num_labeled + self.num_unlabeled]

        # 创建有标签数据集
        labeled_views = [view[labeled_indices] for view in all_views_data]
        labeled_labels = all_labels[labeled_indices]
        labeled_dataset = MultiViewDataset(
            views_data=labeled_views,
            labels=labeled_labels,
            is_labeled=True
        )

        # 创建无标签数据集（标签仅用于评估，训练时不使用）
        unlabeled_views = [view[unlabeled_indices] for view in all_views_data]
        unlabeled_labels = all_labels[unlabeled_indices]  # 保留真实标签用于评估
        unlabeled_dataset = MultiViewDataset(
            views_data=unlabeled_views,
            labels=unlabeled_labels,
            is_labeled=False
        )

        # 创建测试数据集（使用部分数据作为测试集）
        test_size = 500
        test_views = [view[:test_size] for view in all_views_data]
        test_labels = all_labels[:test_size]
        test_dataset = MultiViewDataset(
            views_data=test_views,
            labels=test_labels,
            is_labeled=True
        )

        print(f"数据集加载完成（模拟数据）:")
        print(f"  有标签数据: {len(labeled_dataset)} 样本")
        print(f"  无标签数据: {len(unlabeled_dataset)} 样本")
        print(f"  测试数据: {len(test_dataset)} 样本")
        print(f"  视图数量: {self.num_views}")
        print(f"  视图维度: {self.view_dims}")

        return labeled_dataset, unlabeled_dataset, test_dataset

    def _load_real_data(self):
        """加载真实的HW数据集"""
        import os
        import json

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download the HW dataset or use use_mock_data=True for testing."
            )

        # 加载元数据
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"数据集元数据: {metadata}")

        # 加载所有视图的数据
        all_views_data = []
        for view_name in self.view_names:
            view_path = os.path.join(self.data_dir, f'{view_name}.npy')
            if not os.path.exists(view_path):
                raise FileNotFoundError(f"View data not found: {view_path}")

            view_data = np.load(view_path).astype(np.float32)
            all_views_data.append(view_data)
            print(f"加载视图 {view_name}: shape={view_data.shape}")

        # 加载标签
        labels_path = os.path.join(self.data_dir, 'labels.npy')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        all_labels = np.load(labels_path)
        print(f"加载标签: shape={all_labels.shape}, unique labels={np.unique(all_labels)}")

        # 验证数据一致性
        total_samples = len(all_labels)
        for i, view_data in enumerate(all_views_data):
            assert len(view_data) == total_samples, \
                f"View {self.view_names[i]} has {len(view_data)} samples, but labels have {total_samples}"

        # 设置随机种子
        np.random.seed(self.random_seed)

        # 划分数据集：labeled, unlabeled, test
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # 计算划分点
        test_size = min(500, total_samples // 5)  # 测试集大小：最多500或总数的20%
        train_size = total_samples - test_size

        # 确保有足够的数据
        if train_size < self.num_labeled + self.num_unlabeled:
            print(f"警告: 请求的样本数 ({self.num_labeled + self.num_unlabeled}) "
                  f"超过可用训练样本数 ({train_size})，将调整...")
            self.num_labeled = min(self.num_labeled, train_size // 2)
            self.num_unlabeled = train_size - self.num_labeled

        # 划分索引
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        labeled_indices = train_indices[:self.num_labeled]
        unlabeled_indices = train_indices[self.num_labeled:self.num_labeled + self.num_unlabeled]

        # 创建有标签数据集
        labeled_views = [view[labeled_indices] for view in all_views_data]
        labeled_labels = all_labels[labeled_indices]
        labeled_dataset = MultiViewDataset(
            views_data=labeled_views,
            labels=labeled_labels,
            is_labeled=True
        )

        # 创建无标签数据集（保留真实标签用于评估）
        unlabeled_views = [view[unlabeled_indices] for view in all_views_data]
        unlabeled_labels = all_labels[unlabeled_indices]
        unlabeled_dataset = MultiViewDataset(
            views_data=unlabeled_views,
            labels=unlabeled_labels,
            is_labeled=False
        )

        # 创建测试数据集
        test_views = [view[test_indices] for view in all_views_data]
        test_labels = all_labels[test_indices]
        test_dataset = MultiViewDataset(
            views_data=test_views,
            labels=test_labels,
            is_labeled=True
        )

        print(f"\n数据集加载完成（真实数据）:")
        print(f"  有标签数据: {len(labeled_dataset)} 样本")
        print(f"  无标签数据: {len(unlabeled_dataset)} 样本")
        print(f"  测试数据: {len(test_dataset)} 样本")
        print(f"  视图数量: {self.num_views}")
        print(f"  类别数量: {self.num_classes}")

        # 打印每个数据集的标签分布
        print(f"\n标签分布:")
        print(f"  有标签数据: {np.bincount(labeled_labels)}")
        print(f"  无标签数据: {np.bincount(unlabeled_labels)}")
        print(f"  测试数据: {np.bincount(test_labels)}")

        return labeled_dataset, unlabeled_dataset, test_dataset


class SemiSupervisedSampler(Sampler):
    """
    半监督采样器
    在每个批次中同时采样有标签和无标签数据
    """
    def __init__(self,
                 labeled_dataset: Dataset,
                 unlabeled_dataset: Dataset,
                 batch_size: int,
                 mu: int = 7,
                 shuffle: bool = True):
        """
        Args:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            batch_size: 批次大小（有标签数据）
            mu: 无标签数据与有标签数据的比例
            shuffle: 是否打乱数据
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.mu = mu
        self.shuffle = shuffle

        self.num_labeled = len(labeled_dataset)
        self.num_unlabeled = len(unlabeled_dataset)

        # 每个epoch的迭代次数
        self.num_iterations = self.num_labeled // batch_size

    def __iter__(self):
        """生成采样索引"""
        # 生成有标签数据索引
        labeled_indices = list(range(self.num_labeled))
        if self.shuffle:
            random.shuffle(labeled_indices)

        # 生成无标签数据索引
        unlabeled_indices = list(range(self.num_unlabeled))
        if self.shuffle:
            random.shuffle(unlabeled_indices)

        # 为每个batch生成索引
        for i in range(self.num_iterations):
            # 有标签数据索引
            labeled_batch = labeled_indices[i * self.batch_size:(i + 1) * self.batch_size]

            # 无标签数据索引（可能需要循环）
            unlabeled_batch_size = self.batch_size * self.mu
            unlabeled_batch = []
            for j in range(unlabeled_batch_size):
                idx = (i * unlabeled_batch_size + j) % self.num_unlabeled
                unlabeled_batch.append(unlabeled_indices[idx])

            # 返回 (labeled_indices, unlabeled_indices)
            yield labeled_batch, unlabeled_batch

    def __len__(self):
        return self.num_iterations


class SemiSupervisedDataLoader:
    """
    半监督数据加载器
    在每个批次中同时提供有标签数据(X)和无标签数据(U)
    """
    def __init__(self,
                 labeled_dataset: MultiViewDataset,
                 unlabeled_dataset: MultiViewDataset,
                 batch_size: int = 32,
                 mu: int = 7,
                 shuffle: bool = True,
                 num_workers: int = 0):
        """
        Args:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            batch_size: 批次大小（有标签数据）
            mu: 无标签数据与有标签数据的比例
            shuffle: 是否打乱数据
            num_workers: DataLoader工作线程数
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.mu = mu
        self.shuffle = shuffle
        self.num_workers = num_workers

        # 创建数据加载器
        self.labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size * mu,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True
        )

        # 创建迭代器
        self.labeled_iter = None
        self.unlabeled_iter = None

    def __iter__(self):
        """迭代器"""
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)
        return self

    def __next__(self):
        """获取下一个批次"""
        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            # 有标签数据用完，epoch结束
            raise StopIteration

        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            # 无标签数据用完，重新开始（因为unlabeled数据通常更多）
            self.unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch = next(self.unlabeled_iter)

        return {
            'labeled': labeled_batch,
            'unlabeled': unlabeled_batch
        }

    def __len__(self):
        """返回迭代次数"""
        return len(self.labeled_loader)


def create_hw_dataloaders(
    num_labeled: int = 100,
    num_unlabeled: int = 1900,
    batch_size: int = 32,
    mu: int = 7,
    use_mock_data: bool = True,
    random_seed: int = 42
) -> Tuple[SemiSupervisedDataLoader, DataLoader, HandwrittenNumeralsDataset]:
    """
    创建HW数据集的数据加载器

    Args:
        num_labeled: 有标签样本数量
        num_unlabeled: 无标签样本数量
        batch_size: 批次大小
        mu: 无标签/有标签比例
        use_mock_data: 是否使用模拟数据
        random_seed: 随机种子

    Returns:
        train_loader: 训练数据加载器（包含有标签和无标签数据）
        test_loader: 测试数据加载器
        dataset_info: 数据集信息
    """
    # 创建数据集
    hw_dataset = HandwrittenNumeralsDataset(
        num_labeled=num_labeled,
        num_unlabeled=num_unlabeled,
        random_seed=random_seed
    )

    # 加载数据
    labeled_dataset, unlabeled_dataset, test_dataset = hw_dataset.load_data(
        use_mock_data=use_mock_data
    )

    # 创建训练数据加载器
    train_loader = SemiSupervisedDataLoader(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        batch_size=batch_size,
        mu=mu,
        shuffle=True
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False
    )

    return train_loader, test_loader, hw_dataset


class GenericMultiViewDataset:
    """
    通用多视图数据集加载器
    支持 .mat 格式的多视图数据集
    """
    def __init__(self, data_path: str, random_seed: int = 42):
        """
        Args:
            data_path: .mat 文件路径
            random_seed: 随机种子
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.dataset_name = os.path.basename(data_path).replace('.mat', '')

        # 加载数据
        print(f"正在加载 {data_path}...")
        mat_data = sio.loadmat(data_path)

        # 提取标签（处理不同的标签键名）- 先提取标签，用于判断样本数
        if 'Y' in mat_data:
            self.Y = mat_data['Y'].flatten()
        elif 'gt' in mat_data:
            self.Y = mat_data['gt'].flatten()
        elif 'gnd' in mat_data:
            self.Y = mat_data['gnd'].flatten()
        else:
            raise ValueError(f"找不到标签数据，可用的键: {list(mat_data.keys())}")

        num_samples = len(self.Y)

        # 提取视图数据（处理不同的数据集格式）
        if 'X' in mat_data:
            self.X = mat_data['X']

            # 检查 X 的结构
            if self.X.shape[0] == 1 and len(self.X.shape) == 2:
                # 格式3: X.shape = (1, num_views)，例如 CUB, COIL20
                # X[0] 是一个包含多个视图的数组
                views_array = self.X[0]
                self.num_views = len(views_array)
                self.views_data = []
                for i in range(self.num_views):
                    view = views_array[i].astype(np.float32)
                    # 根据标签数量判断是否需要转置
                    if view.shape[1] == num_samples and view.shape[0] != num_samples:
                        # (特征, 样本) -> (样本, 特征)
                        view = view.T
                    self.views_data.append(view)
            else:
                # 格式1: X 是一个数组，每行包含一个视图
                self.num_views = self.X.shape[0]
                self.views_data = []
                for i in range(self.num_views):
                    view = self.X[i][0].astype(np.float32)
                    self.views_data.append(view)
        else:
            # 格式2: X1, X2, X3, ... (例如 Caltech-5V)
            view_keys = sorted([k for k in mat_data.keys() if k.startswith('X') and k[1:].isdigit()])
            self.num_views = len(view_keys)
            self.views_data = []
            for key in view_keys:
                view = mat_data[key].astype(np.float32)
                self.views_data.append(view)

        self.num_samples = self.views_data[0].shape[0]
        self.num_classes = len(np.unique(self.Y))
        self.view_dims = [view.shape[1] for view in self.views_data]

        print(f"数据集加载完成:")
        print(f"  数据集名称: {self.dataset_name}")
        print(f"  样本数量: {self.num_samples}")
        print(f"  类别数量: {self.num_classes}")
        print(f"  视图数量: {self.num_views}")
        print(f"  视图维度: {self.view_dims}")

        # 标签转换为 0-based（如果标签不是从0开始）
        if self.Y.min() > 0:
            self.Y = self.Y - self.Y.min()

    def load_data(self, labeled_ratio=0.1, test_ratio=0.2):
        """
        划分数据集为有标签、无标签和测试集

        Args:
            labeled_ratio: 有标签数据比例（相对于训练集）
            test_ratio: 测试集比例（相对于总数据）

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


def create_generic_dataloaders(
    data_path: str,
    labeled_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    mu: int = 7,
    random_seed: int = 42
) -> Tuple[SemiSupervisedDataLoader, DataLoader, GenericMultiViewDataset]:
    """
    创建通用多视图数据集的数据加载器

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
        dataset: GenericMultiViewDataset 对象
    """
    # 加载数据集
    dataset = GenericMultiViewDataset(data_path, random_seed)

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
