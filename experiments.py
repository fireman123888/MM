"""
MMatch 批量实验脚本
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

from models import MultiViewEncoder, DistributionAlignment
from data import create_dataloaders
from utils import generate_pseudo_labels


# 配置
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


def train_and_evaluate(mat_path, unlabel_ratio, seed):
    """训练并评估一次实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc, no_improve = 0.0, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            labeled_views = [v.to(device) for v in batch['labeled']['views']]
            labels = batch['labeled']['label'].to(device)
            unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

            optimizer.zero_grad()
            labeled_outputs = model(labeled_views)

            loss_x = sum(criterion(labeled_outputs['view_logits'][v], labels).mean()
                         for v in range(model.num_views))
            if model.use_global_head:
                loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_views)
                view_probs = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                              for v in range(model.num_views)]

            pseudo_targets, masks = generate_pseudo_labels(view_probs, da_modules, TEMPERATURE, THRESHOLD)

            loss_u = 0
            if any(m.sum().item() > 0 for m in masks):
                unlabeled_outputs_train = model(unlabeled_views)
                for v in range(model.num_views):
                    if masks[v].sum() > 0:
                        loss_u += (criterion(unlabeled_outputs_train['view_logits'][v],
                                             pseudo_targets[v]) * masks[v]).mean()
                if model.use_global_head and masks[-1].sum() > 0:
                    loss_u += (criterion(unlabeled_outputs_train['global_logits'],
                                         pseudo_targets[-1]) * masks[-1]).mean()

            loss = loss_x + LAMBDA_U * loss_u
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

        if acc > best_acc + 0.001:
            best_acc, no_improve = acc, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    f1 = f1_score(all_labels, all_preds, average='macro')
    return best_acc, f1


def main():
    print("=" * 70)
    print("MMatch 批量实验")
    print("=" * 70)

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
                acc, f1 = train_and_evaluate(mat_path, unlabel_ratio, seed)
                accs.append(acc)
                f1s.append(f1)

            label_ratio = (1 - unlabel_ratio) * 100
            print(f"  {label_ratio:.0f}% labels: Acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}, "
                  f"F1={np.mean(f1s)*100:.2f}±{np.std(f1s)*100:.2f}")


if __name__ == "__main__":
    main()
