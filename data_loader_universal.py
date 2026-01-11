"""
通用多视图数据加载器
支持 data_new 文件夹中的所有 .mat 数据集
处理不同的数据格式、标签键名、维度顺序
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split


class MultiViewDataset(Dataset):
    """多视图数据集"""

    def __init__(self, views, labels, is_labeled=True):
        """
        Args:
            views: list of numpy arrays, 每个视图的特征 [(n_samples, dim_v), ...]
            labels: numpy array, 标签 (n_samples,)
            is_labeled: bool, 是否为有标签数据
        """
        self.views = [torch.FloatTensor(v) for v in views]
        self.labels = torch.LongTensor(labels)
        self.is_labeled = is_labeled
        self.num_views = len(views)
        self.num_samples = len(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        views = [v[idx] for v in self.views]
        return {
            'views': views,
            'label': self.labels[idx],
            'is_labeled': self.is_labeled
        }


def load_mat_dataset(mat_path):
    """
    加载 .mat 数据集，自动处理不同格式

    Returns:
        views: list of numpy arrays, 每个视图 (n_samples, dim)
        labels: numpy array, 标签 (n_samples,), 从0开始
        view_dims: list of int, 每个视图的维度
        num_classes: int, 类别数
    """
    data = sio.loadmat(mat_path)
    filename = os.path.basename(mat_path)

    # 1. 提取视图数据
    views = []

    if filename == 'Caltech-5V.mat':
        # Caltech-5V: X1, X2, X3, X4, X5 分开存储
        for i in range(1, 6):
            view = data[f'X{i}'].astype(np.float32)
            views.append(view)
    else:
        # 其他数据集: X[0, i] object数组
        X = data['X']
        num_views = X.shape[1]
        for i in range(num_views):
            view = X[0, i].astype(np.float32)
            # 检查是否需要转置 (cub数据集特殊处理)
            if filename == 'cub_googlenet_doc2vec_c10.mat':
                # cub 数据集特征和样本维度是反的
                view = view.T
            views.append(view)

    # 2. 提取标签
    if 'Y' in data:
        labels = data['Y'].flatten()
    elif 'gt' in data:
        labels = data['gt'].flatten()
    elif 'gnd' in data:
        labels = data['gnd'].flatten()
    else:
        raise KeyError(f"Cannot find label key in {filename}")

    # 3. 标签归一化到从0开始
    labels = labels - labels.min()
    labels = labels.astype(np.int64)

    # 4. 验证数据一致性
    n_samples = len(labels)
    for i, v in enumerate(views):
        if v.shape[0] != n_samples:
            raise ValueError(f"View {i+1} has {v.shape[0]} samples, expected {n_samples}")

    view_dims = [v.shape[1] for v in views]
    num_classes = len(np.unique(labels))

    print(f"  数据集: {filename}")
    print(f"  样本数: {n_samples}")
    print(f"  视图数: {len(views)}")
    print(f"  视图维度: {view_dims}")
    print(f"  类别数: {num_classes}")

    return views, labels, view_dims, num_classes


def create_semi_supervised_split(views, labels, test_ratio=0.2, unlabel_ratio=0.9, random_seed=42):
    """
    创建半监督学习的数据划分

    划分策略:
    1. 先划分 80% 训练集 / 20% 测试集
    2. 训练集再划分为 有标签 / 无标签 (根据 unlabel_ratio)

    Args:
        views: list of numpy arrays
        labels: numpy array
        test_ratio: 测试集比例 (默认 0.2)
        unlabel_ratio: 训练集中无标签数据的比例 (如 0.9 表示 90% 无标签)
        random_seed: 随机种子

    Returns:
        train_labeled_views, train_labeled_labels: 有标签训练数据
        train_unlabeled_views, train_unlabeled_labels: 无标签训练数据
        test_views, test_labels: 测试数据
    """
    np.random.seed(random_seed)
    n_samples = len(labels)
    indices = np.arange(n_samples)

    # 第一次划分: 训练集 / 测试集
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed, stratify=labels
    )

    # 第二次划分: 有标签 / 无标签
    train_labels = labels[train_idx]
    num_classes = len(np.unique(train_labels))
    num_labeled = int(len(train_idx) * (1 - unlabel_ratio))

    # 如果有标签样本数小于类别数，确保每个类至少有一个样本
    if num_labeled < num_classes:
        print(f"  警告: 有标签样本数({num_labeled}) < 类别数({num_classes})，每类至少选1个样本")
        # 每个类选择一个样本
        labeled_idx_list = []
        for c in range(num_classes):
            class_indices = np.where(train_labels == c)[0]
            if len(class_indices) > 0:
                np.random.seed(random_seed + c)
                selected = np.random.choice(class_indices, 1)
                labeled_idx_list.append(selected[0])
        labeled_idx = np.array(labeled_idx_list)
        unlabeled_idx = np.array([i for i in range(len(train_idx)) if i not in labeled_idx])
    else:
        # 正常分层采样
        labeled_idx, unlabeled_idx = train_test_split(
            np.arange(len(train_idx)),
            test_size=unlabel_ratio,
            random_state=random_seed,
            stratify=train_labels
        )

    # 映射回原始索引
    labeled_idx = train_idx[labeled_idx]
    unlabeled_idx = train_idx[unlabeled_idx]

    # 划分各视图数据
    train_labeled_views = [v[labeled_idx] for v in views]
    train_labeled_labels = labels[labeled_idx]

    train_unlabeled_views = [v[unlabeled_idx] for v in views]
    train_unlabeled_labels = labels[unlabeled_idx]

    test_views = [v[test_idx] for v in views]
    test_labels = labels[test_idx]

    print(f"  划分结果:")
    print(f"    有标签训练: {len(train_labeled_labels)} ({100*(1-unlabel_ratio):.1f}%)")
    print(f"    无标签训练: {len(train_unlabeled_labels)} ({100*unlabel_ratio:.1f}%)")
    print(f"    测试集: {len(test_labels)} ({100*test_ratio:.1f}%)")

    return (train_labeled_views, train_labeled_labels,
            train_unlabeled_views, train_unlabeled_labels,
            test_views, test_labels)


class SemiSupervisedSampler:
    """半监督采样器，每个batch包含有标签和无标签数据"""

    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size, mu=7):
        """
        Args:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            batch_size: 有标签数据的batch size
            mu: 无标签数据与有标签数据的比例
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.mu = mu

    def __iter__(self):
        labeled_indices = np.random.permutation(len(self.labeled_dataset))
        unlabeled_indices = np.random.permutation(len(self.unlabeled_dataset))

        num_batches = len(labeled_indices) // self.batch_size
        unlabeled_batch_size = self.batch_size * self.mu

        for i in range(num_batches):
            labeled_batch_idx = labeled_indices[i * self.batch_size:(i + 1) * self.batch_size]

            # 无标签数据循环采样
            start = (i * unlabeled_batch_size) % len(unlabeled_indices)
            end = start + unlabeled_batch_size
            if end <= len(unlabeled_indices):
                unlabeled_batch_idx = unlabeled_indices[start:end]
            else:
                # 循环
                unlabeled_batch_idx = np.concatenate([
                    unlabeled_indices[start:],
                    unlabeled_indices[:end - len(unlabeled_indices)]
                ])

            yield labeled_batch_idx, unlabeled_batch_idx

    def __len__(self):
        return len(self.labeled_dataset) // self.batch_size


def collate_semi_supervised(labeled_batch, unlabeled_batch):
    """合并有标签和无标签数据为一个batch"""
    num_views = len(labeled_batch[0]['views'])

    # 处理有标签数据
    labeled_views = [
        torch.stack([sample['views'][v] for sample in labeled_batch])
        for v in range(num_views)
    ]
    labeled_labels = torch.stack([sample['label'] for sample in labeled_batch])

    # 处理无标签数据
    unlabeled_views = [
        torch.stack([sample['views'][v] for sample in unlabeled_batch])
        for v in range(num_views)
    ]
    unlabeled_labels = torch.stack([sample['label'] for sample in unlabeled_batch])

    return {
        'labeled': {'views': labeled_views, 'label': labeled_labels},
        'unlabeled': {'views': unlabeled_views, 'label': unlabeled_labels}
    }


class SemiSupervisedDataLoader:
    """半监督数据加载器"""

    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size=32, mu=7):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.mu = mu
        self.sampler = SemiSupervisedSampler(labeled_dataset, unlabeled_dataset, batch_size, mu)

    def __iter__(self):
        for labeled_idx, unlabeled_idx in self.sampler:
            labeled_batch = [self.labeled_dataset[i] for i in labeled_idx]
            unlabeled_batch = [self.unlabeled_dataset[i] for i in unlabeled_idx]
            yield collate_semi_supervised(labeled_batch, unlabeled_batch)

    def __len__(self):
        return len(self.sampler)


def create_dataloaders(mat_path, test_ratio=0.2, unlabel_ratio=0.9,
                       batch_size=32, mu=7, random_seed=42):
    """
    创建数据加载器

    Args:
        mat_path: .mat 文件路径
        test_ratio: 测试集比例
        unlabel_ratio: 无标签数据比例
        batch_size: 批次大小
        mu: 无标签/有标签比例
        random_seed: 随机种子

    Returns:
        train_loader: 半监督训练数据加载器
        test_loader: 测试数据加载器
        info: dict, 数据集信息
    """
    print(f"\n加载数据集: {mat_path}")

    # 加载数据
    views, labels, view_dims, num_classes = load_mat_dataset(mat_path)

    # 创建数据划分
    (train_labeled_views, train_labeled_labels,
     train_unlabeled_views, train_unlabeled_labels,
     test_views, test_labels) = create_semi_supervised_split(
        views, labels, test_ratio, unlabel_ratio, random_seed
    )

    # 创建数据集
    labeled_dataset = MultiViewDataset(train_labeled_views, train_labeled_labels, is_labeled=True)
    unlabeled_dataset = MultiViewDataset(train_unlabeled_views, train_unlabeled_labels, is_labeled=False)
    test_dataset = MultiViewDataset(test_views, test_labels, is_labeled=True)

    # 创建数据加载器
    train_loader = SemiSupervisedDataLoader(
        labeled_dataset, unlabeled_dataset, batch_size=batch_size, mu=mu
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: {
            'views': [torch.stack([s['views'][v] for s in batch]) for v in range(len(views))],
            'label': torch.stack([s['label'] for s in batch])
        }
    )

    info = {
        'num_views': len(views),
        'view_dims': view_dims,
        'num_classes': num_classes,
        'num_labeled': len(train_labeled_labels),
        'num_unlabeled': len(train_unlabeled_labels),
        'num_test': len(test_labels),
        'dataset_name': os.path.basename(mat_path).replace('.mat', '')
    }

    return train_loader, test_loader, info


if __name__ == "__main__":
    # 测试所有数据集
    data_dir = "data_new"
    datasets = [
        'Caltech101-20.mat',
        'Caltech-5V.mat',
        'COIL20.mat',
        'cub_googlenet_doc2vec_c10.mat',
        'Out_Scene.mat',
        'WebKB.mat',
    ]

    for ds in datasets:
        mat_path = os.path.join(data_dir, ds)
        try:
            train_loader, test_loader, info = create_dataloaders(
                mat_path, test_ratio=0.2, unlabel_ratio=0.9,
                batch_size=32, mu=7, random_seed=42
            )
            print(f"  [OK] 成功加载\n")

            # 测试一个batch
            for batch in train_loader:
                print(f"  Batch test:")
                print(f"    Labeled views: {[v.shape for v in batch['labeled']['views']]}")
                print(f"    Unlabeled views: {[v.shape for v in batch['unlabeled']['views']]}")
                break
            print()
        except Exception as e:
            print(f"  [ERROR] 错误: {e}\n")
