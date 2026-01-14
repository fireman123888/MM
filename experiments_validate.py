"""
验证实验：在所有数据集上验证注意力融合和对比损失的效果
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

# 配置
DATA_DIR = "data_new"
SEEDS = [41, 42, 43, 44, 45]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = [
    "Caltech101-20.mat",
    "Caltech-5V.mat",
    "COIL20.mat",
    "cub_googlenet_doc2vec_c10.mat",
    "Out_Scene.mat",
    "WebKB.mat"
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
    def __init__(self, view_dims, hidden=256, latent=128, n_classes=10, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
            for d in view_dims
        ])
        if use_attention:
            self.attn = nn.Sequential(nn.Linear(latent, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Linear(latent, n_classes)

    def forward(self, views):
        feats = [enc(v) for enc, v in zip(self.encoders, views)]
        stack = torch.stack(feats, dim=1)
        if self.use_attention:
            w = F.softmax(self.attn(stack).squeeze(-1), dim=1)
            fused = (stack * w.unsqueeze(-1)).sum(dim=1)
        else:
            fused = stack.mean(dim=1)
        return self.classifier(fused), fused, feats


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


def add_noise(views, idx, ratio):
    noisy = [v.copy() for v in views]
    v = noisy[idx]
    noisy[idx] = v + np.random.randn(*v.shape).astype(np.float32) * v.std() * ratio
    return noisy


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


def contrastive_loss(feat, labels, temp=0.5):
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


def train_eval(views, labels, config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)

    unlabel_ratio = config.get('unlabel_ratio', 0.9)
    n_labeled = max(1, int(len(tr_idx) * (1 - unlabel_ratio)))
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
    model = Encoder(dims, 256, 128, n_cls, config.get('use_attention', False)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    labeled_mask = np.zeros(len(tr_idx), dtype=bool)
    labeled_mask[labeled_idx] = True

    for _ in range(100):
        model.train()
        for bi, (bv, bl) in enumerate(tr_loader):
            bv = [v.to(DEVICE) for v in bv]
            bl = bl.to(DEVICE)
            s, e = bi*64, min((bi+1)*64, len(tr_idx))
            mask = torch.tensor(labeled_mask[s:e], device=DEVICE)

            opt.zero_grad()
            logits, fused, _ = model(bv)

            if mask.any():
                loss = F.cross_entropy(logits[mask], bl[mask])
                if config.get('use_contrastive') and mask.sum() > 1:
                    loss = loss + 0.1 * contrastive_loss(fused[mask], bl[mask])
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

    return accuracy_score(true, preds)*100, f1_score(true, preds, average='macro')*100


def run_experiment(name, transform_fn, plugins, ratios):
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"{'='*60}")

    results = {}
    for ds in DATASETS:
        ds_name = ds.replace('.mat', '')
        results[ds_name] = {}
        print(f"\n数据集: {ds_name}")

        views_orig, labels_orig = load_data(os.path.join(DATA_DIR, ds))

        for ratio in ratios:
            results[ds_name][ratio] = {}

            for pname, pcfg in plugins.items():
                accs = []
                for seed in SEEDS:
                    np.random.seed(seed)
                    views, labels = transform_fn(views_orig, labels_orig)
                    cfg = {**pcfg, 'unlabel_ratio': ratio}
                    try:
                        acc, _ = train_eval(views, labels, cfg, seed)
                        accs.append(acc)
                    except Exception as e:
                        pass

                if accs:
                    m, s = np.mean(accs), np.std(accs)
                    results[ds_name][ratio][pname] = (m, s)
                    print(f"  {int((1-ratio)*100)}%标签, {pname}: {m:.2f}±{s:.2f}")

    return results


def main():
    print("="*60)
    print("验证实验：注意力融合 & 对比损失")
    print("="*60)
    print(f"设备: {DEVICE}")

    # 实验1: 噪声视图 + 注意力融合
    print("\n\n" + "="*60)
    print("实验1: 噪声视图场景 (50%噪声)")
    print("="*60)

    def noise_transform(v, l):
        return add_noise(v, 0, 0.5), l

    noise_results = run_experiment(
        "噪声视图",
        noise_transform,
        {'Baseline': {}, '+注意力': {'use_attention': True}},
        [0.9, 0.85, 0.8]
    )

    # 实验2: 类别不平衡 + 对比损失
    print("\n\n" + "="*60)
    print("实验2: 类别不平衡场景 (10:1)")
    print("="*60)

    def imbalance_transform(v, l):
        return make_imbalanced(v, l, 10)

    imbalance_results = run_experiment(
        "类别不平衡",
        imbalance_transform,
        {'Baseline': {}, '+对比损失': {'use_contrastive': True}},
        [0.9, 0.85, 0.8]
    )

    # 汇总
    print("\n\n" + "="*60)
    print("结果汇总")
    print("="*60)

    print("\n【噪声视图场景 - 注意力融合效果】")
    print("-"*50)
    for ds, ratios in noise_results.items():
        improvements = []
        for r, plugins in ratios.items():
            base = plugins.get('Baseline', (0,0))[0]
            attn = plugins.get('+注意力', (0,0))[0]
            improvements.append(attn - base)
        avg_imp = np.mean(improvements)
        print(f"  {ds}: 平均提升 {avg_imp:+.2f}%")

    print("\n【类别不平衡场景 - 对比损失效果】")
    print("-"*50)
    for ds, ratios in imbalance_results.items():
        improvements = []
        for r, plugins in ratios.items():
            base = plugins.get('Baseline', (0,0))[0]
            cl = plugins.get('+对比损失', (0,0))[0]
            improvements.append(cl - base)
        avg_imp = np.mean(improvements)
        print(f"  {ds}: 平均提升 {avg_imp:+.2f}%")

    print("\n验证完成！")


if __name__ == "__main__":
    main()
