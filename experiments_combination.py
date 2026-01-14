"""
MMatch 插件组合实验
测试多个插件同时使用是否能叠加效果
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Optional

from models import MultiViewEncoder, DistributionAlignment
from data import create_dataloaders
from utils import generate_pseudo_labels


# ==================== 插件实现 ====================

class LabelDrivenContrastiveLoss(nn.Module):
    """标签驱动对比损失"""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, features.t()) / self.temperature

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(device)
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos * (1 - mask_self)

        pos_count = mask_pos.sum(dim=1)
        if pos_count.sum() == 0:
            return torch.tensor(0.0, device=device)

        exp_sim = torch.exp(similarity) * (1 - mask_self)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (pos_count + 1e-8)

        valid_mask = pos_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        return -mean_log_prob[valid_mask].mean()


class CrossViewConsistencyLoss(nn.Module):
    """跨视图一致性约束损失"""
    def __init__(self, loss_type: str = 'cosine'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        num_views = len(encodings)
        if num_views < 2:
            return torch.tensor(0.0, device=encodings[0].device)

        device = encodings[0].device
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                cos_sim = F.cosine_similarity(encodings[i], encodings[j], dim=1)
                total_loss += (1 - cos_sim).mean()
                num_pairs += 1

        return total_loss / max(num_pairs, 1)


class CSCRContrastiveLoss(nn.Module):
    """CSCR类别驱动对比损失"""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, features.t()) / self.temperature

        # 软一致性矩阵
        if probs is not None:
            probs_norm = F.normalize(probs, dim=1)
            consistency = torch.mm(probs_norm, probs_norm.t())
            consistency = (consistency + 1) / 2
        else:
            labels_col = labels.view(-1, 1)
            consistency = (labels_col == labels_col.t()).float()

        mask_self = torch.eye(batch_size, device=device)
        consistency = consistency * (1 - mask_self)

        pos_count = consistency.sum(dim=1)
        if pos_count.sum() == 0:
            return torch.tensor(0.0, device=device)

        exp_sim = torch.exp(similarity) * (1 - mask_self)
        neg_weight = 1 - consistency
        weighted_exp_neg = (exp_sim * neg_weight).sum(dim=1, keepdim=True)
        log_prob = similarity - torch.log(torch.exp(similarity) + weighted_exp_neg + 1e-8)
        weighted_log_prob = (consistency * log_prob).sum(dim=1) / (pos_count + 1e-8)

        valid_mask = pos_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        return -weighted_log_prob[valid_mask].mean()


# ==================== 配置 ====================

SEEDS = [41, 42, 43, 44, 45]
UNLABEL_RATIOS = [0.99, 0.95, 0.9, 0.85, 0.8]
DATASETS = [
    'Caltech101-20.mat', 'Caltech-5V.mat', 'COIL20.mat',
    'cub_googlenet_doc2vec_c10.mat', 'Out_Scene.mat', 'WebKB.mat',
]
DATA_DIR = 'data_new'

BATCH_SIZE = 32
MU = 7
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20
LAMBDA_U = 1.0
THRESHOLD = 0.95
TEMPERATURE = 0.5

# 组合配置
COMBINATIONS = {
    'CL+CV': {'contrastive': True, 'consistency': True, 'cscr': False,
              'lambda_cl': 0.1, 'lambda_cv': 0.1, 'lambda_cscr': 0.0},
    'CL+CSCR': {'contrastive': True, 'consistency': False, 'cscr': True,
                'lambda_cl': 0.1, 'lambda_cv': 0.0, 'lambda_cscr': 0.1},
    'CV+CSCR': {'contrastive': False, 'consistency': True, 'cscr': True,
                'lambda_cl': 0.0, 'lambda_cv': 0.1, 'lambda_cscr': 0.1},
    'ALL': {'contrastive': True, 'consistency': True, 'cscr': True,
            'lambda_cl': 0.05, 'lambda_cv': 0.05, 'lambda_cscr': 0.05},
}


def train_and_evaluate(mat_path, unlabel_ratio, seed, config):
    """训练并评估一次实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader, test_loader, info = create_dataloaders(
        mat_path, unlabel_ratio=unlabel_ratio,
        batch_size=BATCH_SIZE, mu=MU, random_seed=seed
    )

    model = MultiViewEncoder(
        num_views=info['num_views'],
        input_dims=info['view_dims'],
        hidden_dims=[256, 128],
        encoding_dim=64,
        num_classes=info['num_classes'],
        dropout=0.5,
        use_global_head=True,
        global_hidden_dim=128
    ).to(device)

    da_modules = [DistributionAlignment(info['num_classes']) for _ in range(info['num_views'] + 1)]

    criterion = nn.CrossEntropyLoss(reduction='none')

    # 初始化损失函数
    cl_loss_fn = LabelDrivenContrastiveLoss(TEMPERATURE).to(device) if config['contrastive'] else None
    cv_loss_fn = CrossViewConsistencyLoss('cosine').to(device) if config['consistency'] else None
    cscr_loss_fn = CSCRContrastiveLoss(TEMPERATURE).to(device) if config['cscr'] else None

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc, best_f1, no_improve = 0.0, 0.0, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            labeled_views = [v.to(device) for v in batch['labeled']['views']]
            labels = batch['labeled']['label'].to(device)
            unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

            optimizer.zero_grad()
            labeled_outputs = model(labeled_views)

            # 监督损失
            loss_x = sum(criterion(labeled_outputs['view_logits'][v], labels).mean()
                         for v in range(model.num_views))
            if model.use_global_head:
                loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

            # 标签驱动对比损失
            loss_cl = torch.tensor(0.0, device=device)
            if cl_loss_fn is not None:
                global_rep = labeled_outputs['global_representation']
                loss_cl = cl_loss_fn(global_rep, labels)
                for v in range(model.num_views):
                    loss_cl += cl_loss_fn(labeled_outputs['encodings'][v], labels)
                loss_cl = loss_cl / (model.num_views + 1)

            # 跨视图一致性损失
            loss_cv = torch.tensor(0.0, device=device)
            if cv_loss_fn is not None:
                loss_cv = cv_loss_fn(labeled_outputs['encodings'])

            # CSCR损失
            loss_cscr = torch.tensor(0.0, device=device)
            if cscr_loss_fn is not None:
                with torch.no_grad():
                    labeled_probs = F.softmax(labeled_outputs['global_logits'], dim=1)
                global_rep = labeled_outputs['global_representation']
                loss_cscr = cscr_loss_fn(global_rep, labels, labeled_probs)

            # 无标签数据
            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_views)
                view_probs = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                              for v in range(model.num_views)]

            pseudo_targets, masks = generate_pseudo_labels(view_probs, da_modules, TEMPERATURE, THRESHOLD)

            loss_u = torch.tensor(0.0, device=device)
            if any(m.sum().item() > 0 for m in masks):
                unlabeled_outputs_train = model(unlabeled_views)
                for v in range(model.num_views):
                    if masks[v].sum() > 0:
                        loss_u += (criterion(unlabeled_outputs_train['view_logits'][v],
                                             pseudo_targets[v]) * masks[v]).mean()
                if model.use_global_head and masks[-1].sum() > 0:
                    loss_u += (criterion(unlabeled_outputs_train['global_logits'],
                                         pseudo_targets[-1]) * masks[-1]).mean()

                # 对无标签数据也应用一致性损失
                if cv_loss_fn is not None:
                    loss_cv += cv_loss_fn(unlabeled_outputs_train['encodings'])
                    loss_cv = loss_cv / 2

            # 总损失
            loss = (loss_x + LAMBDA_U * loss_u +
                    config['lambda_cl'] * loss_cl +
                    config['lambda_cv'] * loss_cv +
                    config['lambda_cscr'] * loss_cscr)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                views = [v.to(device) for v in batch['views']]
                outputs = model(views)
                preds = outputs['global_logits'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        if acc > best_acc + 0.001:
            best_acc, best_f1, no_improve = acc, f1, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_acc, best_f1


def main():
    print("=" * 80)
    print("MMatch 插件组合实验")
    print("=" * 80)
    print("\n组合说明:")
    print("  CL+CV   = 对比损失 + 一致性约束")
    print("  CL+CSCR = 对比损失 + CSCR")
    print("  CV+CSCR = 一致性约束 + CSCR")
    print("  ALL     = 对比损失 + 一致性约束 + CSCR")
    print()

    for combo_name, config in COMBINATIONS.items():
        print("=" * 80)
        print(f"组合: {combo_name}")
        print("=" * 80)

        for dataset in DATASETS:
            mat_path = os.path.join(DATA_DIR, dataset)
            if not os.path.exists(mat_path):
                print(f"跳过 {dataset} (文件不存在)")
                continue

            print(f"\n数据集: {dataset}")
            print("-" * 50)

            for unlabel_ratio in UNLABEL_RATIOS:
                accs, f1s = [], []
                for seed in SEEDS:
                    acc, f1 = train_and_evaluate(mat_path, unlabel_ratio, seed, config)
                    accs.append(acc)
                    f1s.append(f1)

                label_ratio = (1 - unlabel_ratio) * 100
                print(f"  {label_ratio:.0f}% labels: Acc={np.mean(accs)*100:.2f}+/-{np.std(accs)*100:.2f}, "
                      f"F1={np.mean(f1s)*100:.2f}+/-{np.std(f1s)*100:.2f}")


if __name__ == "__main__":
    main()
