"""
多视图半监督学习实验脚本
- 20%测试集，80%训练集（二次划分：有标签/无标签）
- 评估指标：Accuracy 和 F1 Score
- 5个种子 [41,42,43,44,45]，计算平均值和标准差
- unlabel比例：0.99, 0.95, 0.9, 0.85, 0.8
- 支持CUDA加速
"""
import os
import random
import statistics
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data_loader import MultiViewDataset, SemiSupervisedDataLoader
from multi_view_encoder import MultiViewEncoder


# ==================== 配置参数 ====================
SEEDS = [41, 42, 43, 44, 45]
UNLABEL_RATIOS = [0.99, 0.95, 0.9, 0.85, 0.8]  # 无标签比例（训练集内）
TEST_RATIO = 0.2  # 测试集比例

DATA_DIR = 'data_new'
DATASETS = [
    'Caltech101-20.mat',
    'Caltech-5V.mat',
    'COIL20.mat',
    'cub_googlenet_doc2vec_c10.mat',
    'Out_Scene.mat',
    'WebKB.mat',
]

# 训练参数
BATCH_SIZE = 32
MU = 7  # 无标签/有标签批次比例
NUM_EPOCHS = 200
LEARNING_RATE = 0.005
PATIENCE = 40

# MMatch 参数
LAMBDA_U = 1.0
THRESHOLD = 0.95
TEMPERATURE = 0.5
DA_MOMENTUM = 0.999


# ==================== 数据加载器 ====================
class GenericDatasetLoader:
    """通用多视图数据集加载器，处理不同格式的.mat文件"""

    def __init__(self, data_path: str, random_seed: int = 42):
        self.data_path = data_path
        self.random_seed = random_seed
        self.dataset_name = os.path.basename(data_path).replace('.mat', '')

        print(f"正在加载 {self.dataset_name}...")
        mat_data = sio.loadmat(data_path)

        # 提取标签（处理不同键名）
        if 'Y' in mat_data:
            self.Y = mat_data['Y'].flatten()
        elif 'gt' in mat_data:
            self.Y = mat_data['gt'].flatten()
        elif 'gnd' in mat_data:
            self.Y = mat_data['gnd'].flatten()
        else:
            raise ValueError(f"找不到标签数据，可用的键: {[k for k in mat_data.keys() if not k.startswith('_')]}")

        num_samples = len(self.Y)

        # 提取视图数据
        if 'X' in mat_data:
            X = mat_data['X']
            if X.shape[0] == 1 and len(X.shape) == 2:
                # X.shape = (1, num_views)
                views_array = X[0]
                self.num_views = len(views_array)
                self.views_data = []
                for i in range(self.num_views):
                    view = views_array[i].astype(np.float32)
                    # 检查是否需要转置
                    if view.shape[0] != num_samples and view.shape[1] == num_samples:
                        view = view.T
                    self.views_data.append(view)
            else:
                # X每行是一个视图
                self.num_views = X.shape[0]
                self.views_data = []
                for i in range(self.num_views):
                    view = X[i][0].astype(np.float32)
                    self.views_data.append(view)
        else:
            # X1, X2, X3, ... 格式
            view_keys = sorted([k for k in mat_data.keys() if k.startswith('X') and k[1:].isdigit()])
            self.num_views = len(view_keys)
            self.views_data = []
            for key in view_keys:
                view = mat_data[key].astype(np.float32)
                self.views_data.append(view)

        # 验证维度
        self.num_samples = self.views_data[0].shape[0]
        for i, view in enumerate(self.views_data):
            assert view.shape[0] == self.num_samples, \
                f"视图{i}样本数{view.shape[0]}与预期{self.num_samples}不符"

        self.num_classes = len(np.unique(self.Y))
        self.view_dims = [view.shape[1] for view in self.views_data]

        # 标签转为0-based
        if self.Y.min() > 0:
            self.Y = self.Y - self.Y.min()

        print(f"  样本数: {self.num_samples}, 类别数: {self.num_classes}")
        print(f"  视图数: {self.num_views}, 维度: {self.view_dims}")

    def load_data(self, unlabel_ratio: float = 0.9, test_ratio: float = 0.2):
        """
        划分数据集
        Args:
            unlabel_ratio: 训练集中无标签数据的比例
            test_ratio: 测试集比例
        """
        np.random.seed(self.random_seed)

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)

        # 划分测试集和训练集
        test_size = int(self.num_samples * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # 训练集中划分有标签和无标签
        train_size = len(train_indices)
        num_unlabeled = int(train_size * unlabel_ratio)
        num_labeled = train_size - num_unlabeled

        labeled_indices = train_indices[:num_labeled]
        unlabeled_indices = train_indices[num_labeled:]

        print(f"  数据划分: 有标签={len(labeled_indices)}, 无标签={len(unlabeled_indices)}, 测试={len(test_indices)}")

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

        return labeled_dataset, unlabeled_dataset, test_dataset


# ==================== 训练模块 ====================
class DistributionAlignment:
    """分布对齐模块"""
    def __init__(self, num_classes, momentum=0.999, device='cpu'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device
        self.q_tilde = (torch.ones(num_classes) / num_classes).to(device)

    def update(self, probs):
        q_batch = probs.mean(dim=0)
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach()

    def align(self, probs):
        q_tilde = torch.clamp(self.q_tilde, min=1e-6)
        scaled = probs / q_tilde.unsqueeze(0)
        return scaled / scaled.sum(dim=1, keepdim=True)


def sharpen(probs, T):
    sharpened = probs ** (1 / T)
    return sharpened / sharpened.sum(dim=1, keepdim=True)


def train_epoch(model, train_loader, criterion, optimizer, device, da_module,
                lambda_u=1.0, threshold=0.95, temperature=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        labeled_views = [v.to(device) for v in batch['labeled']['views']]
        labels = batch['labeled']['label'].to(device)
        unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

        optimizer.zero_grad()
        labeled_outputs = model(labeled_views)

        # 监督损失
        loss_x = 0
        for v in range(model.num_views):
            loss_x += criterion(labeled_outputs['view_logits'][v], labels).mean()
        if model.use_global_head:
            loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

        # 无监督损失
        loss_u = 0
        if model.use_global_head:
            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_views)
                unlabeled_probs = F.softmax(unlabeled_outputs['global_logits'], dim=1)
                da_module.update(unlabeled_probs)
                aligned_probs = da_module.align(unlabeled_probs)
                pseudo_labels = sharpen(aligned_probs, T=temperature)
                max_probs, pseudo_targets = torch.max(pseudo_labels, dim=1)
                mask = max_probs >= threshold

            if mask.sum() > 0:
                unlabeled_outputs_train = model(unlabeled_views)
                for v in range(model.num_views):
                    loss_u += (criterion(unlabeled_outputs_train['view_logits'][v], pseudo_targets) * mask).mean()
                loss_u += (criterion(unlabeled_outputs_train['global_logits'], pseudo_targets) * mask).mean()

        loss = loss_x + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, test_loader, device):
    """评估模型，返回准确率和F1"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            views = [v.to(device) for v in batch['views']]
            labels = batch['label'].to(device)
            outputs = model(views)

            if model.use_global_head:
                preds = torch.argmax(outputs['global_logits'], dim=1)
            else:
                # 融合多视图预测
                logits_sum = sum(outputs['view_logits'])
                preds = torch.argmax(logits_sum, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean() * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    return acc, f1


def run_single_experiment(data_path, unlabel_ratio, seed, device):
    """运行单次实验"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 加载数据
    dataset = GenericDatasetLoader(data_path, random_seed=seed)
    labeled_ds, unlabeled_ds, test_ds = dataset.load_data(
        unlabel_ratio=unlabel_ratio,
        test_ratio=TEST_RATIO
    )

    # 检查有标签数据是否足够
    if len(labeled_ds) < BATCH_SIZE:
        batch_size = max(1, len(labeled_ds) // 2)
    else:
        batch_size = BATCH_SIZE

    # 创建数据加载器
    train_loader = SemiSupervisedDataLoader(
        labeled_dataset=labeled_ds,
        unlabeled_dataset=unlabeled_ds,
        batch_size=batch_size,
        mu=MU,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

    # 创建模型
    model = MultiViewEncoder(
        num_views=dataset.num_views,
        input_dims=dataset.view_dims,
        hidden_dims=[256, 128],
        encoding_dim=64,
        num_classes=dataset.num_classes,
        use_global_head=True
    ).to(device)

    # 优化器和损失
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    da_module = DistributionAlignment(dataset.num_classes, DA_MOMENTUM, device)

    # 训练
    best_acc = 0
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, da_module,
            LAMBDA_U, THRESHOLD, TEMPERATURE
        )
        acc, f1 = evaluate(model, test_loader, device)

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_acc, best_f1


def run_all_experiments():
    """运行所有实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"种子列表: {SEEDS}")
    print(f"无标签比例: {UNLABEL_RATIOS}")
    print("=" * 80)

    # 结果存储
    results = {}

    for dataset_file in DATASETS:
        data_path = os.path.join(DATA_DIR, dataset_file)
        if not os.path.exists(data_path):
            print(f"[跳过] 数据集不存在: {data_path}")
            continue

        dataset_name = dataset_file.replace('.mat', '')
        results[dataset_name] = {}
        print(f"\n{'='*80}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*80}")

        for unlabel_ratio in UNLABEL_RATIOS:
            labeled_ratio = 1 - unlabel_ratio
            print(f"\n--- 有标签比例: {labeled_ratio*100:.0f}% (无标签: {unlabel_ratio*100:.0f}%) ---")

            acc_list = []
            f1_list = []

            for seed in SEEDS:
                print(f"  种子 {seed}: ", end="", flush=True)
                try:
                    acc, f1 = run_single_experiment(data_path, unlabel_ratio, seed, device)
                    acc_list.append(acc)
                    f1_list.append(f1)
                    print(f"Acc={acc:.2f}%, F1={f1:.2f}%")
                except Exception as e:
                    print(f"错误: {e}")
                    continue

            if len(acc_list) >= 2:
                avg_acc = statistics.mean(acc_list)
                std_acc = statistics.stdev(acc_list)
                avg_f1 = statistics.mean(f1_list)
                std_f1 = statistics.stdev(f1_list)
            elif len(acc_list) == 1:
                avg_acc, std_acc = acc_list[0], 0.0
                avg_f1, std_f1 = f1_list[0], 0.0
            else:
                avg_acc, std_acc, avg_f1, std_f1 = 0, 0, 0, 0

            results[dataset_name][unlabel_ratio] = {
                'acc': (avg_acc, std_acc),
                'f1': (avg_f1, std_f1),
                'raw_acc': acc_list,
                'raw_f1': f1_list
            }

            print(f"  >> 结果: Acc={avg_acc:.2f}±{std_acc:.2f}%, F1={avg_f1:.2f}±{std_f1:.2f}%")

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        print("-" * 60)
        print(f"{'Unlabel%':<10} {'Acc (mean±std)':<20} {'F1 (mean±std)':<20}")
        print("-" * 60)
        for unlabel_ratio in UNLABEL_RATIOS:
            if unlabel_ratio in dataset_results:
                r = dataset_results[unlabel_ratio]
                acc_str = f"{r['acc'][0]:.2f}±{r['acc'][1]:.2f}"
                f1_str = f"{r['f1'][0]:.2f}±{r['f1'][1]:.2f}"
                print(f"{unlabel_ratio*100:<10.0f} {acc_str:<20} {f1_str:<20}")

    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"experiment_results_{timestamp}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("实验结果汇总\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"设备: {device}\n")
        f.write(f"种子: {SEEDS}\n")
        f.write(f"无标签比例: {UNLABEL_RATIOS}\n")
        f.write("=" * 80 + "\n\n")

        for dataset_name, dataset_results in results.items():
            f.write(f"\n{dataset_name}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Unlabel%':<10} {'Acc (mean±std)':<20} {'F1 (mean±std)':<20}\n")
            f.write("-" * 60 + "\n")
            for unlabel_ratio in UNLABEL_RATIOS:
                if unlabel_ratio in dataset_results:
                    r = dataset_results[unlabel_ratio]
                    acc_str = f"{r['acc'][0]:.2f}±{r['acc'][1]:.2f}"
                    f1_str = f"{r['f1'][0]:.2f}±{r['f1'][1]:.2f}"
                    f.write(f"{unlabel_ratio*100:<10.0f} {acc_str:<20} {f1_str:<20}\n")

    print(f"\n结果已保存到: {result_file}")
    return results


if __name__ == '__main__':
    run_all_experiments()
