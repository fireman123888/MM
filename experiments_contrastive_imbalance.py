"""
验证实验：对比损失在类别不平衡场景下的效果
在所有数据集上测试不同不平衡比例
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
SEEDS = [41, 42, 43]  # 使用3个种子加速
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = [
    "Caltech101-20.mat",
    "Caltech-5V.mat",
    "COIL20.mat",
    "cub_googlenet_doc2vec_c10.mat",
    "Out_Scene.mat",
    "WebKB.mat"
]

IMBALANCE_RATIOS = [5, 10, 20]  # 不平衡比例
LABEL_RATIOS = [0.9, 0.85, 0.8]  # 无标签比例


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
    """创建不平衡数据"""
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


def train_eval(views, labels, use_contrastive, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)

    unlabel_ratio = 0.9
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
    model = Encoder(dims, 256, 128, n_cls).to(DEVICE)
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
            logits, fused = model(bv)

            if mask.any():
                loss = F.cross_entropy(logits[mask], bl[mask])
                if use_contrastive and mask.sum() > 1:
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

    return accuracy_score(true, preds) * 100


def main():
    print("=" * 60)
    print("验证：对比损失在类别不平衡场景下的效果")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"不平衡比例: {IMBALANCE_RATIOS}")
    print(f"数据集: {len(DATASETS)}个")

    all_results = {}

    for ds in DATASETS:
        ds_name = ds.replace('.mat', '')
        print(f"\n{'='*50}")
        print(f"数据集: {ds_name}")
        print(f"{'='*50}")

        views_orig, labels_orig = load_data(os.path.join(DATA_DIR, ds))
        print(f"原始样本数: {len(labels_orig)}, 类别数: {len(np.unique(labels_orig))}")

        all_results[ds_name] = {}

        for imb_ratio in IMBALANCE_RATIOS:
            print(f"\n不平衡比例 {imb_ratio}:1")
            all_results[ds_name][imb_ratio] = {}

            baseline_accs = []
            contrastive_accs = []

            for seed in SEEDS:
                np.random.seed(seed)
                views, labels = make_imbalanced(views_orig.copy(), labels_orig.copy(), imb_ratio)

                try:
                    base_acc = train_eval(views, labels, False, seed)
                    cl_acc = train_eval(views, labels, True, seed)
                    baseline_accs.append(base_acc)
                    contrastive_accs.append(cl_acc)
                except Exception as e:
                    print(f"  错误: {e}")

            if baseline_accs:
                base_m, base_s = np.mean(baseline_accs), np.std(baseline_accs)
                cl_m, cl_s = np.mean(contrastive_accs), np.std(contrastive_accs)
                diff = cl_m - base_m

                all_results[ds_name][imb_ratio] = {
                    'baseline': (base_m, base_s),
                    'contrastive': (cl_m, cl_s),
                    'diff': diff
                }

                sign = "+" if diff > 0 else ""
                print(f"  Baseline:    {base_m:.2f}±{base_s:.2f}")
                print(f"  +对比损失:   {cl_m:.2f}±{cl_s:.2f} ({sign}{diff:.2f}%)")

    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    print("\n各数据集在不同不平衡比例下的提升:")
    print("-" * 50)
    print(f"{'数据集':<25} {'5:1':<12} {'10:1':<12} {'20:1':<12}")
    print("-" * 50)

    effective_count = 0
    total_count = 0

    for ds_name, ratios in all_results.items():
        row = f"{ds_name:<25}"
        for imb_ratio in IMBALANCE_RATIOS:
            if imb_ratio in ratios and ratios[imb_ratio]:
                diff = ratios[imb_ratio]['diff']
                sign = "+" if diff > 0 else ""
                row += f" {sign}{diff:.2f}%".ljust(12)
                total_count += 1
                if diff > 0.5:
                    effective_count += 1
            else:
                row += " N/A".ljust(12)
        print(row)

    print("-" * 50)
    print(f"\n有效提升(>0.5%)的比例: {effective_count}/{total_count} = {effective_count/total_count*100:.1f}%")

    # 统计正负
    positive = sum(1 for ds in all_results.values() for r in ds.values() if r and r.get('diff', 0) > 0)
    negative = sum(1 for ds in all_results.values() for r in ds.values() if r and r.get('diff', 0) < 0)
    print(f"正向提升: {positive}次, 负向影响: {negative}次")

    print("\n验证完成！")


if __name__ == "__main__":
    main()
