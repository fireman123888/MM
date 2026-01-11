"""
MMatch 消融实验脚本
验证各组件的贡献：
1. Supervised Only - 只用有标签数据（无伪标签）
2. Global PL - 原始MMatch，只用全局分类器生成伪标签
3. Cross-View Uniform - 互教学，均匀加权（不用置信度）
4. Cross-View Confidence - 完整方法，置信度加权互教学
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader_universal import create_dataloaders
from multi_view_encoder import MultiViewEncoder


# ==================== 配置 ====================
SEEDS = [41, 42, 43, 44, 45]
UNLABEL_RATIOS = [0.99, 0.95]  # 1%和5%标签率
DATASETS = [
    'Caltech101-20.mat',
    'Caltech-5V.mat',
    'COIL20.mat',
    'Out_Scene.mat',
    'WebKB.mat',
]
DATA_DIR = 'data_new'

# 训练参数
BATCH_SIZE = 32
MU = 7
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20

# MMatch 参数
LAMBDA_U = 1.0
THRESHOLD = 0.95
TEMPERATURE = 0.5
DA_MOMENTUM = 0.999

# 消融实验配置 - 只对比视图级伪标签的贡献
ABLATION_METHODS = [
    'global_pl',            # 全局伪标签（原始MMatch）
    'cross_view_confidence' # 视图级伪标签（我们的方法）
]


# ==================== 分布对齐模块 ====================
class DistributionAlignment:
    def __init__(self, num_classes, momentum=0.999):
        self.num_classes = num_classes
        self.momentum = momentum
        self.q_tilde = torch.ones(num_classes) / num_classes

    def update(self, probs):
        q_batch = probs.mean(dim=0)
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach().cpu()

    def align(self, probs):
        q_tilde = self.q_tilde.to(probs.device)
        q_tilde = torch.clamp(q_tilde, min=1e-6)
        scaled = probs / q_tilde.unsqueeze(0)
        aligned_probs = scaled / scaled.sum(dim=1, keepdim=True)
        return aligned_probs


def sharpen(probs, T=0.5):
    sharpened = probs ** (1.0 / T)
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
    return sharpened


# ==================== 方法1: 全局伪标签（原始MMatch） ====================
def generate_global_pseudo_labels(global_probs, da_module, temperature=0.5, threshold=0.95):
    """原始MMatch：只用全局分类器生成伪标签"""
    da_module.update(global_probs)
    aligned_probs = da_module.align(global_probs)
    sharpened_probs = sharpen(aligned_probs, T=temperature)
    max_probs, pseudo_targets = torch.max(sharpened_probs, dim=1)
    mask = max_probs >= threshold
    return pseudo_targets, mask


# ==================== 方法2: 互教学均匀加权 ====================
def generate_cross_view_uniform_labels(view_probs_list, da_modules, temperature=0.5, threshold=0.95):
    """互教学：排除自己，均匀加权平均"""
    num_views = len(view_probs_list)
    pseudo_targets_list = []
    masks_list = []

    for v in range(num_views):
        # 收集其他视图的预测
        other_probs = [view_probs_list[i] for i in range(num_views) if i != v]
        # 均匀平均
        avg_probs = torch.stack(other_probs, dim=0).mean(dim=0)

        da_modules[v].update(avg_probs)
        aligned_probs = da_modules[v].align(avg_probs)
        sharpened_probs = sharpen(aligned_probs, T=temperature)
        max_probs, pseudo_targets = torch.max(sharpened_probs, dim=1)
        mask = max_probs >= threshold

        pseudo_targets_list.append(pseudo_targets)
        masks_list.append(mask)

    # 全局伪标签
    all_avg_probs = torch.stack(view_probs_list, dim=0).mean(dim=0)
    da_modules[-1].update(all_avg_probs)
    aligned_probs_global = da_modules[-1].align(all_avg_probs)
    sharpened_probs_global = sharpen(aligned_probs_global, T=temperature)
    max_probs_global, pseudo_targets_global = torch.max(sharpened_probs_global, dim=1)
    mask_global = max_probs_global >= threshold

    pseudo_targets_list.append(pseudo_targets_global)
    masks_list.append(mask_global)

    return pseudo_targets_list, masks_list


# ==================== 方法3: 互教学置信度加权（完整方法） ====================
def generate_cross_view_confidence_labels(view_probs_list, da_modules, temperature=0.5, threshold=0.95):
    """互教学：排除自己，置信度加权平均"""
    num_views = len(view_probs_list)
    pseudo_targets_list = []
    masks_list = []

    # 计算各视图的置信度
    confidences = [torch.max(probs, dim=1)[0] for probs in view_probs_list]

    for v in range(num_views):
        other_probs = []
        other_confs = []
        for i in range(num_views):
            if i != v:
                other_probs.append(view_probs_list[i])
                other_confs.append(confidences[i])

        weights = torch.stack(other_confs, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        probs_stack = torch.stack(other_probs, dim=0)
        weighted_probs = (probs_stack * weights.unsqueeze(-1)).sum(dim=0)

        da_modules[v].update(weighted_probs)
        aligned_probs = da_modules[v].align(weighted_probs)
        sharpened_probs = sharpen(aligned_probs, T=temperature)
        max_probs, pseudo_targets = torch.max(sharpened_probs, dim=1)
        mask = max_probs >= threshold

        pseudo_targets_list.append(pseudo_targets)
        masks_list.append(mask)

    # 全局伪标签
    weights = torch.stack(confidences, dim=0)
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
    probs_stack = torch.stack(view_probs_list, dim=0)
    weighted_probs_global = (probs_stack * weights.unsqueeze(-1)).sum(dim=0)

    da_modules[-1].update(weighted_probs_global)
    aligned_probs_global = da_modules[-1].align(weighted_probs_global)
    sharpened_probs_global = sharpen(aligned_probs_global, T=temperature)
    max_probs_global, pseudo_targets_global = torch.max(sharpened_probs_global, dim=1)
    mask_global = max_probs_global >= threshold

    pseudo_targets_list.append(pseudo_targets_global)
    masks_list.append(mask_global)

    return pseudo_targets_list, masks_list


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device, da_modules, method='cross_view_confidence'):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        labeled_views = [v.to(device) for v in batch['labeled']['views']]
        labels = batch['labeled']['label'].to(device)
        unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

        optimizer.zero_grad()

        # 有标签数据前向传播
        labeled_outputs = model(labeled_views)

        # 监督损失
        loss_x = 0
        for v in range(model.num_views):
            loss_x += criterion(labeled_outputs['view_logits'][v], labels).mean()
        if model.use_global_head:
            loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

        # ========== 根据方法计算无监督损失 ==========
        loss_u = 0

        if method == 'supervised_only':
            # 方法1: 只用有标签数据，无无监督损失
            pass

        else:
            # 无标签数据前向传播
            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_views)

            view_probs_list = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                              for v in range(model.num_views)]

            if method == 'global_pl':
                # 方法2: 全局伪标签
                global_probs = F.softmax(unlabeled_outputs['global_logits'], dim=1)
                pseudo_targets, mask = generate_global_pseudo_labels(
                    global_probs, da_modules[0], TEMPERATURE, THRESHOLD
                )

                if mask.sum() > 0:
                    unlabeled_outputs_train = model(unlabeled_views)
                    # 所有视图和全局都用同一个伪标签
                    for v in range(model.num_views):
                        view_logits = unlabeled_outputs_train['view_logits'][v]
                        loss_u += (criterion(view_logits, pseudo_targets) * mask).mean()
                    if model.use_global_head:
                        global_logits = unlabeled_outputs_train['global_logits']
                        loss_u += (criterion(global_logits, pseudo_targets) * mask).mean()

            elif method == 'cross_view_uniform':
                # 方法3: 互教学均匀加权
                pseudo_targets_list, masks_list = generate_cross_view_uniform_labels(
                    view_probs_list, da_modules, TEMPERATURE, THRESHOLD
                )

                any_mask = sum([m.sum().item() for m in masks_list]) > 0
                if any_mask:
                    unlabeled_outputs_train = model(unlabeled_views)
                    for v in range(model.num_views):
                        if masks_list[v].sum() > 0:
                            view_logits = unlabeled_outputs_train['view_logits'][v]
                            loss_u += (criterion(view_logits, pseudo_targets_list[v]) * masks_list[v]).mean()
                    if model.use_global_head and masks_list[-1].sum() > 0:
                        global_logits = unlabeled_outputs_train['global_logits']
                        loss_u += (criterion(global_logits, pseudo_targets_list[-1]) * masks_list[-1]).mean()

            elif method == 'cross_view_confidence':
                # 方法4: 互教学置信度加权
                pseudo_targets_list, masks_list = generate_cross_view_confidence_labels(
                    view_probs_list, da_modules, TEMPERATURE, THRESHOLD
                )

                any_mask = sum([m.sum().item() for m in masks_list]) > 0
                if any_mask:
                    unlabeled_outputs_train = model(unlabeled_views)
                    for v in range(model.num_views):
                        if masks_list[v].sum() > 0:
                            view_logits = unlabeled_outputs_train['view_logits'][v]
                            loss_u += (criterion(view_logits, pseudo_targets_list[v]) * masks_list[v]).mean()
                    if model.use_global_head and masks_list[-1].sum() > 0:
                        global_logits = unlabeled_outputs_train['global_logits']
                        loss_u += (criterion(global_logits, pseudo_targets_list[-1]) * masks_list[-1]).mean()

        loss = loss_x + LAMBDA_U * loss_u
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            views = [v.to(device) for v in batch['views']]
            labels = batch['label']
            outputs = model(views)
            preds = torch.argmax(outputs['global_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


# ==================== 单次实验 ====================
def run_single_experiment(mat_path, unlabel_ratio, seed, device, method):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader, test_loader, info = create_dataloaders(
        mat_path, test_ratio=0.2, unlabel_ratio=unlabel_ratio,
        batch_size=BATCH_SIZE, mu=MU, random_seed=seed
    )

    model = MultiViewEncoder(
        num_views=info['num_views'], input_dims=info['view_dims'],
        hidden_dims=[256, 128], encoding_dim=64, num_classes=info['num_classes'],
        dropout=0.5, use_batch_norm=True, use_global_head=True, global_hidden_dim=128
    ).to(device)

    # 创建分布对齐模块
    if method == 'global_pl':
        da_modules = [DistributionAlignment(info['num_classes'], DA_MOMENTUM)]
    else:
        da_modules = [DistributionAlignment(info['num_classes'], DA_MOMENTUM)
                      for _ in range(info['num_views'] + 1)]

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, da_modules, method)
        acc, f1 = evaluate(model, test_loader, device)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_acc, best_f1


# ==================== 主函数 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Ablation Methods: {ABLATION_METHODS}")
    print(f"Seeds: {SEEDS}")
    print(f"Unlabel ratios: {UNLABEL_RATIOS}")
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"ablation_results_{timestamp}.txt"

    # results[unlabel_ratio][dataset_name][method]
    all_results = {}

    for unlabel_ratio in UNLABEL_RATIOS:
        label_ratio = 1 - unlabel_ratio
        print(f"\n{'#'*100}")
        print(f"# UNLABEL RATIO: {unlabel_ratio} (Label Ratio: {label_ratio:.0%})")
        print(f"{'#'*100}")

        all_results[unlabel_ratio] = {}

        for dataset in DATASETS:
            mat_path = os.path.join(DATA_DIR, dataset)
            dataset_name = dataset.replace('.mat', '')
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            all_results[unlabel_ratio][dataset_name] = {}

            for method in ABLATION_METHODS:
                print(f"\n  Method: {method}")
                accs = []
                f1s = []

                for seed in SEEDS:
                    print(f"    Seed {seed}...", end=" ", flush=True)
                    try:
                        acc, f1 = run_single_experiment(mat_path, unlabel_ratio, seed, device, method)
                        accs.append(acc)
                        f1s.append(f1)
                        print(f"ACC={acc:.4f}, F1={f1:.4f}")
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                if len(accs) > 0:
                    acc_mean, acc_std = np.mean(accs), np.std(accs)
                    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
                    all_results[unlabel_ratio][dataset_name][method] = {
                        'acc_mean': acc_mean, 'acc_std': acc_std,
                        'f1_mean': f1_mean, 'f1_std': f1_std
                    }
                    print(f"  => ACC: {acc_mean:.4f} +/- {acc_std:.4f}, F1: {f1_mean:.4f} +/- {f1_std:.4f}")

    # 打印汇总表格
    print("\n" + "=" * 120)
    print("ABLATION STUDY SUMMARY")
    print("=" * 120)

    for unlabel_ratio in UNLABEL_RATIOS:
        label_ratio = 1 - unlabel_ratio
        print(f"\n### Label Ratio: {label_ratio:.0%} (Unlabel: {unlabel_ratio})")
        print("-" * 100)

        header = f"{'Dataset':<20}"
        for method in ABLATION_METHODS:
            header += f" | {method:<25}"
        header += " | Diff"
        print(header)
        print("-" * 100)

        for dataset_name in all_results[unlabel_ratio]:
            row_acc = f"{dataset_name:<20}"
            row_f1 = f"{'(F1)':<20}"

            accs = []
            for method in ABLATION_METHODS:
                if method in all_results[unlabel_ratio][dataset_name]:
                    m = all_results[unlabel_ratio][dataset_name][method]
                    row_acc += f" | {m['acc_mean']:.4f}+/-{m['acc_std']:.4f}     "
                    row_f1 += f" | {m['f1_mean']:.4f}+/-{m['f1_std']:.4f}     "
                    accs.append(m['acc_mean'])
                else:
                    row_acc += f" | {'N/A':<25}"
                    row_f1 += f" | {'N/A':<25}"
                    accs.append(0)

            # 计算差异
            if len(accs) == 2:
                diff = accs[1] - accs[0]
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                row_acc += f" | {diff_str}"

            print(row_acc)
            print(row_f1)
        print()

    # 保存结果
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Unlabel Ratios: {UNLABEL_RATIOS}\n")
        f.write("=" * 100 + "\n\n")

        for unlabel_ratio in UNLABEL_RATIOS:
            label_ratio = 1 - unlabel_ratio
            f.write(f"\n### Label Ratio: {label_ratio:.0%}\n")
            f.write("-" * 50 + "\n")

            for dataset_name in all_results[unlabel_ratio]:
                f.write(f"{dataset_name}:\n")
                for method in ABLATION_METHODS:
                    if method in all_results[unlabel_ratio][dataset_name]:
                        m = all_results[unlabel_ratio][dataset_name][method]
                        f.write(f"  {method}: ACC={m['acc_mean']:.4f}+/-{m['acc_std']:.4f}, ")
                        f.write(f"F1={m['f1_mean']:.4f}+/-{m['f1_std']:.4f}\n")
                f.write("\n")

    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
