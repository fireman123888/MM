"""
挑战场景实验：通过人工构造挑战场景展示各插件优势
"""
import os
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置
DATA_DIR = "data_new"
SEEDS = [41, 42, 43, 44, 45]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleMultiViewDataset(Dataset):
    def __init__(self, views, labels):
        self.views = [torch.FloatTensor(v) for v in views]
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [v[idx] for v in self.views], self.labels[idx]


class SimpleEncoder(nn.Module):
    """简化的多视图编码器"""
    def __init__(self, view_dims, hidden_dim=256, latent_dim=128, num_classes=10, use_attention=False):
        super().__init__()
        self.num_views = len(view_dims)
        self.use_attention = use_attention

        # 每个视图的编码器
        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for dim in view_dims
        ])

        # 注意力融合
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, views):
        # 编码每个视图
        view_features = [enc(v) for enc, v in zip(self.view_encoders, views)]
        view_stack = torch.stack(view_features, dim=1)  # [B, V, D]

        if self.use_attention:
            # 计算注意力权重
            attn_scores = self.attention(view_stack).squeeze(-1)  # [B, V]
            attn_weights = F.softmax(attn_scores, dim=1)
            fused = (view_stack * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            fused = view_stack.mean(dim=1)

        logits = self.classifier(fused)
        return {'logits': logits, 'fused': fused, 'view_features': view_features}


def load_dataset(mat_path):
    """加载数据集"""
    data = sio.loadmat(mat_path)
    filename = os.path.basename(mat_path)

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

    for key in ['Y', 'gt', 'gnd']:
        if key in data:
            labels = data[key].flatten()
            break

    labels = (labels - labels.min()).astype(np.int64)
    return views, labels


def add_noise_to_view(views, noise_view_idx, noise_ratio):
    """给指定视图添加噪声"""
    noisy_views = [v.copy() for v in views]
    v = noisy_views[noise_view_idx]
    noise = np.random.randn(*v.shape).astype(np.float32) * v.std() * noise_ratio
    noisy_views[noise_view_idx] = v + noise
    return noisy_views


def create_imbalanced_data(views, labels, imbalance_ratio):
    """创建不平衡数据"""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    keep_idx = []
    for i, cls in enumerate(unique_labels):
        cls_idx = np.where(labels == cls)[0]
        if i < n_classes // 2:
            keep_idx.extend(cls_idx)
        else:
            n_keep = max(3, len(cls_idx) // imbalance_ratio)
            keep_idx.extend(np.random.choice(cls_idx, n_keep, replace=False))

    keep_idx = np.array(keep_idx)
    return [v[keep_idx] for v in views], labels[keep_idx]


def add_label_noise(labels, noise_ratio):
    """添加标签噪声"""
    noisy = labels.copy()
    n_noise = int(len(labels) * noise_ratio)
    noise_idx = np.random.choice(len(labels), n_noise, replace=False)
    unique = np.unique(labels)
    for idx in noise_idx:
        other = unique[unique != labels[idx]]
        noisy[idx] = np.random.choice(other)
    return noisy


def contrastive_loss(features, labels, temperature=0.5):
    """对比损失"""
    features = F.normalize(features, dim=1)
    similarity = torch.mm(features, features.t()) / temperature

    labels = labels.view(-1, 1)
    mask = (labels == labels.t()).float()

    # 排除对角线
    mask_diag = torch.eye(len(labels), device=features.device)
    mask = mask - mask_diag

    exp_sim = torch.exp(similarity) * (1 - mask_diag)
    pos = (exp_sim * mask).sum(dim=1)
    neg = exp_sim.sum(dim=1)

    loss = -torch.log(pos / (neg + 1e-8) + 1e-8)
    loss = loss[mask.sum(dim=1) > 0].mean()
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=features.device)


def consistency_loss(view_features):
    """一致性损失"""
    loss = 0
    count = 0
    for i in range(len(view_features)):
        for j in range(i + 1, len(view_features)):
            loss = loss + F.mse_loss(view_features[i], view_features[j])
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0)


def cscr_loss(features, labels, num_classes, temperature=0.1):
    """CSCR损失"""
    features = F.normalize(features, dim=1)

    # 构建标签一致性矩阵
    labels_onehot = F.one_hot(labels, num_classes).float()
    consistency_matrix = torch.mm(labels_onehot, labels_onehot.t())

    # 对比损失
    similarity = torch.mm(features, features.t()) / temperature
    exp_sim = torch.exp(similarity)

    # 加权
    pos = (exp_sim * consistency_matrix).sum(dim=1) - exp_sim.diag()
    neg = exp_sim.sum(dim=1) - exp_sim.diag()

    loss = -torch.log(pos / (neg + 1e-8) + 1e-8)
    return loss.mean()


def train_and_evaluate(views, labels, config, seed):
    """训练并评估"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 划分数据
    n_samples = len(labels)
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
    train_labels = labels[train_idx]

    # 划分有标签/无标签
    unlabel_ratio = config.get('unlabel_ratio', 0.9)
    n_labeled = max(1, int(len(train_idx) * (1 - unlabel_ratio)))

    labeled_idx = []
    for cls in np.unique(train_labels):
        cls_mask = train_labels == cls
        cls_indices = np.where(cls_mask)[0]
        n_per_class = max(1, n_labeled // len(np.unique(train_labels)))
        if len(cls_indices) > 0:
            selected = np.random.choice(cls_indices, min(n_per_class, len(cls_indices)), replace=False)
            labeled_idx.extend(selected)

    labeled_idx = np.array(labeled_idx[:n_labeled])

    train_views = [v[train_idx] for v in views]
    train_labels_all = labels[train_idx]
    test_views = [v[test_idx] for v in views]
    test_labels = labels[test_idx]

    # 标签噪声
    if config.get('label_noise'):
        train_labels_all = add_label_noise(train_labels_all, config['label_noise'])

    # 创建数据集
    train_dataset = SimpleMultiViewDataset(train_views, train_labels_all)
    test_dataset = SimpleMultiViewDataset(test_views, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 模型
    view_dims = [v.shape[1] for v in views]
    num_classes = len(np.unique(labels))
    model = SimpleEncoder(view_dims, 256, 128, num_classes, config.get('use_attention', False))
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 有标签样本的mask
    labeled_mask_full = np.zeros(len(train_idx), dtype=bool)
    labeled_mask_full[labeled_idx] = True

    # 训练
    for epoch in range(100):
        model.train()
        for batch_idx, (batch_views, batch_labels) in enumerate(train_loader):
            batch_views = [v.to(DEVICE) for v in batch_views]
            batch_labels = batch_labels.to(DEVICE)

            # 计算当前batch中哪些是有标签的
            start_idx = batch_idx * 64
            end_idx = min(start_idx + 64, len(train_idx))
            batch_labeled = torch.tensor(labeled_mask_full[start_idx:end_idx], device=DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_views)

            # 分类损失
            if batch_labeled.any():
                loss = F.cross_entropy(outputs['logits'][batch_labeled], batch_labels[batch_labeled])
            else:
                loss = torch.tensor(0.0, device=DEVICE)

            # 额外损失
            if config.get('use_contrastive') and batch_labeled.any() and batch_labeled.sum() > 1:
                cl_loss = contrastive_loss(outputs['fused'][batch_labeled], batch_labels[batch_labeled])
                loss = loss + config.get('lambda_cl', 0.1) * cl_loss

            if config.get('use_consistency'):
                cv_loss = consistency_loss(outputs['view_features'])
                loss = loss + config.get('lambda_cv', 0.1) * cv_loss

            if config.get('use_cscr') and batch_labeled.any() and batch_labeled.sum() > 1:
                cs_loss = cscr_loss(outputs['fused'][batch_labeled], batch_labels[batch_labeled], num_classes)
                loss = loss + config.get('lambda_cscr', 0.1) * cs_loss

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

    # 评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_views, batch_labels in test_loader:
            batch_views = [v.to(DEVICE) for v in batch_views]
            preds = model(batch_views)['logits'].argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    return acc, f1


def run_scenario(name, dataset, transform_fn, plugins, unlabel_ratios):
    """运行单个场景"""
    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"数据集: {dataset}")
    print(f"{'='*60}")

    mat_path = os.path.join(DATA_DIR, dataset)
    views_orig, labels_orig = load_dataset(mat_path)

    results = {}
    for ratio in unlabel_ratios:
        results[ratio] = {}
        print(f"\n标签率: {int((1-ratio)*100)}%")

        for plugin_name, plugin_config in plugins.items():
            accs, f1s = [], []
            for seed in SEEDS:
                np.random.seed(seed)
                views, labels = transform_fn(views_orig.copy(), labels_orig.copy())
                config = {**plugin_config, 'unlabel_ratio': ratio}

                try:
                    acc, f1 = train_and_evaluate(views, labels, config, seed)
                    accs.append(acc)
                    f1s.append(f1)
                except Exception as e:
                    print(f"  错误 ({plugin_name}): {e}")

            if accs:
                mean_acc, std_acc = np.mean(accs), np.std(accs)
                results[ratio][plugin_name] = (mean_acc, std_acc)
                print(f"  {plugin_name}: {mean_acc:.2f}±{std_acc:.2f}")

    return results


def main():
    print("=" * 60)
    print("挑战场景实验")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    all_results = {}

    # 场景1: 噪声视图 → 注意力融合
    print("\n\n" + "=" * 60)
    print("场景1: 噪声视图 (50%高斯噪声)")
    print("预期: 注意力融合能自动降低噪声视图权重")
    print("=" * 60)

    def add_noise(views, labels):
        return add_noise_to_view(views, 0, 0.5), labels

    noise_results = run_scenario(
        "噪声视图", "Caltech-5V.mat", add_noise,
        {'Baseline': {}, '+注意力融合': {'use_attention': True}},
        [0.95, 0.9, 0.85]
    )
    all_results['噪声视图'] = noise_results

    # 场景2: 类别不平衡 → 对比损失
    print("\n\n" + "=" * 60)
    print("场景2: 类别不平衡 (10:1)")
    print("预期: 对比损失能增强少数类的表示学习")
    print("=" * 60)

    def make_imbalanced(views, labels):
        return create_imbalanced_data(views, labels, 10)

    imbalance_results = run_scenario(
        "类别不平衡", "Caltech101-20.mat", make_imbalanced,
        {'Baseline': {}, '+对比损失': {'use_contrastive': True, 'lambda_cl': 0.1}},
        [0.9, 0.85, 0.8]
    )
    all_results['类别不平衡'] = imbalance_results

    # 场景3: 极低标签率 → 一致性约束
    print("\n\n" + "=" * 60)
    print("场景3: 极低标签率")
    print("预期: 一致性约束利用无标签数据提升性能")
    print("=" * 60)

    def identity(views, labels):
        return views, labels

    low_label_results = run_scenario(
        "极低标签率", "COIL20.mat", identity,
        {'Baseline': {}, '+一致性约束': {'use_consistency': True, 'lambda_cv': 0.1}},
        [0.98, 0.97, 0.95]
    )
    all_results['极低标签率'] = low_label_results

    # 场景4: 标签噪声 → CSCR
    print("\n\n" + "=" * 60)
    print("场景4: 标签噪声 (20%)")
    print("预期: CSCR的软一致性矩阵能容忍噪声标签")
    print("=" * 60)

    label_noise_results = run_scenario(
        "标签噪声", "Caltech101-20.mat", identity,
        {'Baseline': {}, '+CSCR': {'use_cscr': True, 'lambda_cscr': 0.1, 'label_noise': 0.2}},
        [0.9, 0.85, 0.8]
    )
    all_results['标签噪声'] = label_noise_results

    # 汇总
    print("\n\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    for scenario, results in all_results.items():
        print(f"\n{scenario}:")
        for ratio, plugins in results.items():
            baseline = plugins.get('Baseline', (0, 0))[0]
            for name, (acc, std) in plugins.items():
                if name != 'Baseline':
                    improvement = acc - baseline
                    print(f"  {int((1-ratio)*100)}%标签, {name}: {acc:.2f}% (提升{improvement:+.2f}%)")

    print("\n实验完成！")


if __name__ == "__main__":
    main()
