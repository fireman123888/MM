"""
通用多视图数据加载器
"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MultiViewDataset(Dataset):
    """多视图数据集"""

    def __init__(self, views, labels, is_labeled=True):
        self.views = [torch.FloatTensor(v) for v in views]
        self.labels = torch.LongTensor(labels)
        self.is_labeled = is_labeled
        self.num_views = len(views)
        self.num_samples = len(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'views': [v[idx] for v in self.views],
            'label': self.labels[idx],
            'is_labeled': self.is_labeled
        }


def load_mat_dataset(mat_path):
    """加载 .mat 数据集，自动处理不同格式"""
    data = sio.loadmat(mat_path)
    filename = os.path.basename(mat_path)

    # 提取视图数据
    views = []
    if filename == 'Caltech-5V.mat':
        for i in range(1, 6):
            views.append(data[f'X{i}'].astype(np.float32))
    else:
        X = data['X']
        for i in range(X.shape[1]):
            view = X[0, i].astype(np.float32)
            if filename == 'cub_googlenet_doc2vec_c10.mat':
                view = view.T
            views.append(view)

    # 提取标签
    for key in ['Y', 'gt', 'gnd']:
        if key in data:
            labels = data[key].flatten()
            break
    else:
        raise KeyError(f"Cannot find label key in {filename}")

    labels = (labels - labels.min()).astype(np.int64)

    # 验证一致性
    n_samples = len(labels)
    for i, v in enumerate(views):
        if v.shape[0] != n_samples:
            raise ValueError(f"View {i+1} has {v.shape[0]} samples, expected {n_samples}")

    return views, labels, [v.shape[1] for v in views], len(np.unique(labels))


def create_semi_supervised_split(views, labels, test_ratio=0.2, unlabel_ratio=0.9, random_seed=42):
    """创建半监督学习的数据划分"""
    np.random.seed(random_seed)
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed, stratify=labels
    )

    train_labels = labels[train_idx]
    num_classes = len(np.unique(train_labels))
    num_labeled = int(len(train_idx) * (1 - unlabel_ratio))

    if num_labeled < num_classes:
        labeled_idx_list = []
        for c in range(num_classes):
            class_indices = np.where(train_labels == c)[0]
            if len(class_indices) > 0:
                np.random.seed(random_seed + c)
                labeled_idx_list.append(np.random.choice(class_indices, 1)[0])
        labeled_idx = np.array(labeled_idx_list)
        unlabeled_idx = np.array([i for i in range(len(train_idx)) if i not in labeled_idx])
    else:
        labeled_idx, unlabeled_idx = train_test_split(
            np.arange(len(train_idx)), test_size=unlabel_ratio,
            random_state=random_seed, stratify=train_labels
        )

    labeled_idx = train_idx[labeled_idx]
    unlabeled_idx = train_idx[unlabeled_idx]

    return (
        [v[labeled_idx] for v in views], labels[labeled_idx],
        [v[unlabeled_idx] for v in views], labels[unlabeled_idx],
        [v[test_idx] for v in views], labels[test_idx]
    )


class SemiSupervisedSampler:
    """半监督采样器"""

    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size, mu=7):
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

            start = (i * unlabeled_batch_size) % len(unlabeled_indices)
            end = start + unlabeled_batch_size
            if end <= len(unlabeled_indices):
                unlabeled_batch_idx = unlabeled_indices[start:end]
            else:
                unlabeled_batch_idx = np.concatenate([
                    unlabeled_indices[start:],
                    unlabeled_indices[:end - len(unlabeled_indices)]
                ])

            yield labeled_batch_idx, unlabeled_batch_idx

    def __len__(self):
        return len(self.labeled_dataset) // self.batch_size


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
            yield self._collate(labeled_batch, unlabeled_batch)

    def __len__(self):
        return len(self.sampler)

    @staticmethod
    def _collate(labeled_batch, unlabeled_batch):
        num_views = len(labeled_batch[0]['views'])
        return {
            'labeled': {
                'views': [torch.stack([s['views'][v] for s in labeled_batch]) for v in range(num_views)],
                'label': torch.stack([s['label'] for s in labeled_batch])
            },
            'unlabeled': {
                'views': [torch.stack([s['views'][v] for s in unlabeled_batch]) for v in range(num_views)],
                'label': torch.stack([s['label'] for s in unlabeled_batch])
            }
        }


def create_dataloaders(mat_path, test_ratio=0.2, unlabel_ratio=0.9,
                       batch_size=32, mu=7, random_seed=42):
    """创建数据加载器"""
    views, labels, view_dims, num_classes = load_mat_dataset(mat_path)

    (train_labeled_views, train_labeled_labels,
     train_unlabeled_views, train_unlabeled_labels,
     test_views, test_labels) = create_semi_supervised_split(
        views, labels, test_ratio, unlabel_ratio, random_seed
    )

    labeled_dataset = MultiViewDataset(train_labeled_views, train_labeled_labels, True)
    unlabeled_dataset = MultiViewDataset(train_unlabeled_views, train_unlabeled_labels, False)
    test_dataset = MultiViewDataset(test_views, test_labels, True)

    train_loader = SemiSupervisedDataLoader(labeled_dataset, unlabeled_dataset, batch_size, mu)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: {
            'views': [torch.stack([s['views'][v] for s in batch]) for v in range(len(views))],
            'label': torch.stack([s['label'] for s in batch])
        }
    )

    return train_loader, test_loader, {
        'num_views': len(views),
        'view_dims': view_dims,
        'num_classes': num_classes,
        'num_labeled': len(train_labeled_labels),
        'num_unlabeled': len(train_unlabeled_labels),
        'num_test': len(test_labels),
        'dataset_name': os.path.basename(mat_path).replace('.mat', '')
    }
