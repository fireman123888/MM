"""
优化对比损失：解决极端类别不平衡问题
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = "data_new"
SEEDS = [41, 42, 43]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = [
    "Caltech101-20.mat",
    "Caltech-5V.mat",
    "cub_googlenet_doc2vec_c10.mat",
    "Out_Scene.mat",
]

class SimpleDataset(Dataset):
    def __init__(self, views, labels):
        self.views = [torch.FloatTensor(v) for v in views]
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [v[idx] for v in self.views], self.labels[idx]


class Encoder(nn.Module):
    def __init__(self, view_dims, hidden=256, latent=128, n_classes=10):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
            for d in view_dims
        ])
        self.classifier = nn.Linear(latent, n_classes)

    def forward(self, views):
        feats = [enc(v) for enc, v in zip(self.encoders, views)]
        fused = torch.stack(feats, dim=1).mean(dim=1)
        return self.classifier(fused), fused


def load_data(path):
    data = sio.loadmat(path)
    name = os.path.basename(path)
    views = []
    if name == 'Caltech-5V.mat':
        for i in range(1, 6):
            views.append(data[f'X{i}'].astype(np.float32))
    else:
        X = data['X']
        for i in range(X.shape[1]):
            v = X[0, i].astype(np.float32)
            if name == 'cub_googlenet_doc2vec_c10.mat':
                v = v.T
            views.append(v)
    for k in ['Y', 'gt', 'gnd']:
        if k in data:
            labels = data[k].flatten()
            break
    labels = (labels - labels.min()).astype(np.int64)
    return views, labels


def make_imbalanced(views, labels, ratio):
    unique = np.unique(labels)
    n = len(unique)
    keep = []
    for i, c in enumerate(unique):
        idx = np.where(labels == c)[0]
        if i < n // 2:
            keep.extend(idx)
        else:
            keep.extend(np.random.choice(idx, max(3, len(idx)//ratio), replace=False))
    keep = np.array(keep)
    return [v[keep] for v in views], labels[keep]


# ============== 对比损失变体 ==============

def contrastive_loss_vanilla(feat, labels, temp=0.5):
    """原始对比损失（基线）"""
    feat = F.normalize(feat, dim=1)
    sim = torch.mm(feat, feat.t()) / temp
    mask = (labels.view(-1,1) == labels.view(1,-1)).float()
    diag = torch.eye(len(labels), device=feat.device)
    mask = mask - diag
    exp = torch.exp(sim) * (1 - diag)
    pos = (exp * mask).sum(1)
    neg = exp.sum(1)
    loss = -torch.log(pos / (neg + 1e-8) + 1e-8)
    loss = loss[mask.sum(1) > 0].mean()
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=feat.device)


def contrastive_loss_balanced(feat, labels, temp=0.5):
    """优化1: 类别平衡加权对比损失"""
    feat = F.normalize(feat, dim=1)
    sim = torch.mm(feat, feat.t()) / temp

    # 计算每个类别的样本数，用于加权
    unique_labels = torch.unique(labels)
    class_counts = torch.zeros(len(unique_labels), device=feat.device)
    for i, c in enumerate(unique_labels):
        class_counts[i] = (labels == c).sum().float()

    # 计算每个样本的权重（类别频率的倒数）
    sample_weights = torch.zeros(len(labels), device=feat.device)
    for i, c in enumerate(unique_labels):
        mask_c = (labels == c)
        sample_weights[mask_c] = 1.0 / class_counts[i]
    sample_weights = sample_weights / sample_weights.sum() * len(labels)

    mask = (labels.view(-1,1) == labels.view(1,-1)).float()
    diag = torch.eye(len(labels), device=feat.device)
    mask = mask - diag
    exp = torch.exp(sim) * (1 - diag)
    pos = (exp * mask).sum(1)
    neg = exp.sum(1)

    loss = -torch.log(pos / (neg + 1e-8) + 1e-8)
    loss = loss * sample_weights  # 加权
    loss = loss[mask.sum(1) > 0].mean()
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=feat.device)


def contrastive_loss_prototype(feat, labels, temp=0.5):
    """优化2: 原型对比损失 - 使用类别中心而非样本对"""
    feat = F.normalize(feat, dim=1)
    unique_labels = torch.unique(labels)

    # 计算每个类别的原型（中心）
    prototypes = []
    for c in unique_labels:
        mask_c = (labels == c)
        proto = feat[mask_c].mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)  # [C, D]
    prototypes = F.normalize(prototypes, dim=1)

    # 每个样本与所有原型的相似度
    sim = torch.mm(feat, prototypes.t()) / temp  # [N, C]

    # 构建目标
    targets = torch.zeros(len(labels), dtype=torch.long, device=feat.device)
    for i, c in enumerate(unique_labels):
        targets[labels == c] = i

    loss = F.cross_entropy(sim, targets)
    return loss


def contrastive_loss_hard_mining(feat, labels, temp=0.5):
    """优化3: 难样本挖掘对比损失"""
    feat = F.normalize(feat, dim=1)
    sim = torch.mm(feat, feat.t()) / temp
    mask = (labels.view(-1,1) == labels.view(1,-1)).float()
    diag = torch.eye(len(labels), device=feat.device)
    mask = mask - diag
    neg_mask = 1 - mask - diag

    exp = torch.exp(sim)

    # 难正样本：同类中相似度最低的
    pos_sim = sim * mask + (1 - mask) * 1e9
    hard_pos = pos_sim.min(dim=1)[0]

    # 难负样本：异类中相似度最高的
    neg_sim = sim * neg_mask - (1 - neg_mask) * 1e9
    hard_neg = neg_sim.max(dim=1)[0]

    # 三元组损失风格
    loss = F.relu(hard_neg - hard_pos + 0.5)
    valid = mask.sum(1) > 0
    loss = loss[valid].mean()
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=feat.device)


# 损失函数映射
LOSS_FUNCTIONS = {
    'vanilla': contrastive_loss_vanilla,
    'balanced': contrastive_loss_balanced,
    'prototype': contrastive_loss_prototype,
    'hard_mining': contrastive_loss_hard_mining,
}


def train_eval(views, labels, loss_type, seed):
    """训练并评估"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)

    n_labeled = max(1, int(len(tr_idx) * 0.1))
    tr_labels = labels[tr_idx]

    labeled_idx = []
    for c in np.unique(tr_labels):
        c_idx = np.where(tr_labels == c)[0]
        n = max(1, n_labeled // len(np.unique(tr_labels)))
        labeled_idx.extend(np.random.choice(c_idx, min(n, len(c_idx)), replace=False))
    labeled_idx = np.array(labeled_idx[:n_labeled])

    tr_views = [v[tr_idx] for v in views]
    te_views = [v[te_idx] for v in views]

    tr_ds = SimpleDataset(tr_views, labels[tr_idx])
    te_ds = SimpleDataset(te_views, labels[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=64)

    dims = [v.shape[1] for v in views]
    n_cls = len(np.unique(labels))
    model = Encoder(dims, 256, 128, n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    labeled_mask = np.zeros(len(tr_idx), dtype=bool)
    labeled_mask[labeled_idx] = True
    loss_fn = LOSS_FUNCTIONS.get(loss_type) if loss_type else None

    for _ in range(100):
        model.train()
        for bi, (bv, bl) in enumerate(tr_loader):
            bv = [v.to(DEVICE) for v in bv]
            bl = bl.to(DEVICE)
            s, e = bi*64, min((bi+1)*64, len(tr_idx))
            mask = torch.tensor(labeled_mask[s:e], device=DEVICE)

            opt.zero_grad()
            logits, fused = model(bv)

            if mask.any():
                loss = F.cross_entropy(logits[mask], bl[mask])
                if loss_fn and mask.sum() > 1:
                    loss = loss + 0.1 * loss_fn(fused[mask], bl[mask])
            else:
                loss = torch.tensor(0.0, device=DEVICE)

            if loss.requires_grad:
                loss.backward()
                opt.step()

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for bv, bl in te_loader:
            bv = [v.to(DEVICE) for v in bv]
            p = model(bv)[0].argmax(1).cpu()
            preds.extend(p.numpy())
            true.extend(bl.numpy())

    return accuracy_score(true, preds) * 100


def main():
    print("=" * 60)
    print("优化对比损失：解决极端类别不平衡问题")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    methods = ['baseline', 'vanilla', 'balanced', 'prototype', 'hard_mining']
    imb_ratio = 20  # 极端不平衡

    all_results = {}

    for ds in DATASETS:
        ds_name = ds.replace('.mat', '')
        print(f"\n{'='*50}")
        print(f"数据集: {ds_name} (不平衡比例 {imb_ratio}:1)")
        print(f"{'='*50}")

        views_orig, labels_orig = load_data(os.path.join(DATA_DIR, ds))
        all_results[ds_name] = {}

        for method in methods:
            accs = []
            loss_type = None if method == 'baseline' else method

            for seed in SEEDS:
                np.random.seed(seed)
                views, labels = make_imbalanced(
                    [v.copy() for v in views_orig],
                    labels_orig.copy(),
                    imb_ratio
                )
                try:
                    acc = train_eval(views, labels, loss_type, seed)
                    accs.append(acc)
                except Exception as e:
                    print(f"  {method} 错误: {e}")

            if accs:
                m, s = np.mean(accs), np.std(accs)
                all_results[ds_name][method] = (m, s)
                print(f"  {method:<12}: {m:.2f}±{s:.2f}")

    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总 (相对baseline的提升)")
    print("=" * 60)

    print(f"\n{'数据集':<25} {'vanilla':<12} {'balanced':<12} {'prototype':<12} {'hard_mining':<12}")
    print("-" * 75)

    for ds_name, results in all_results.items():
        base = results.get('baseline', (0,0))[0]
        row = f"{ds_name:<25}"
        for method in ['vanilla', 'balanced', 'prototype', 'hard_mining']:
            if method in results:
                diff = results[method][0] - base
                sign = "+" if diff > 0 else ""
                row += f" {sign}{diff:.2f}%".ljust(12)
            else:
                row += " N/A".ljust(12)
        print(row)

    print("\n验证完成！")


if __name__ == "__main__":
    main()
