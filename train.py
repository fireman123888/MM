"""
MMatch 训练脚本
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models import MultiViewEncoder, DistributionAlignment
from data import create_dataloaders
from utils import compute_accuracy, generate_pseudo_labels


def train_epoch(model, train_loader, criterion, optimizer, device,
                da_modules, lambda_u=1.0, threshold=0.95, temperature=0.5):
    """训练一个epoch"""
    model.train()
    total_loss, total_loss_x, total_loss_u = 0, 0, 0
    total_global_acc, total_mask_ratio = 0.0, 0.0
    num_batches = 0

    for batch in train_loader:
        labeled_views = [v.to(device) for v in batch['labeled']['views']]
        labels = batch['labeled']['label'].to(device)
        unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

        optimizer.zero_grad()

        # 有标签数据前向传播
        labeled_outputs = model(labeled_views)

        # 监督损失
        loss_x = sum(criterion(labeled_outputs['view_logits'][v], labels).mean()
                     for v in range(model.num_views))
        if model.use_global_head:
            loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

        # 无标签数据生成伪标签
        with torch.no_grad():
            unlabeled_outputs = model(unlabeled_views)
            view_probs_list = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                               for v in range(model.num_views)]

        pseudo_targets_list, masks_list = generate_pseudo_labels(
            view_probs_list, da_modules, temperature, threshold
        )

        mask_ratio = sum(m.float().mean().item() for m in masks_list) / len(masks_list)

        # 无监督损失
        loss_u = 0
        if any(m.sum().item() > 0 for m in masks_list):
            unlabeled_outputs_train = model(unlabeled_views)
            for v in range(model.num_views):
                if masks_list[v].sum() > 0:
                    loss_u += (criterion(unlabeled_outputs_train['view_logits'][v],
                                         pseudo_targets_list[v]) * masks_list[v]).mean()
            if model.use_global_head and masks_list[-1].sum() > 0:
                loss_u += (criterion(unlabeled_outputs_train['global_logits'],
                                     pseudo_targets_list[-1]) * masks_list[-1]).mean()

        loss = loss_x + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_x += loss_x.item()
        if isinstance(loss_u, torch.Tensor):
            total_loss_u += loss_u.item()
        if model.use_global_head:
            total_global_acc += compute_accuracy(labeled_outputs['global_logits'], labels).item()
        total_mask_ratio += mask_ratio
        num_batches += 1

    return (total_loss / num_batches, total_loss_x / num_batches,
            total_loss_u / num_batches, total_global_acc / num_batches,
            total_mask_ratio / num_batches)


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss, total_global_acc = 0, 0.0
    total_view_acc = [0.0] * model.num_views
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            views = [v.to(device) for v in batch['views']]
            labels = batch['label'].to(device)
            outputs = model(views)

            loss = sum(criterion(outputs['view_logits'][v], labels) for v in range(model.num_views))
            for v in range(model.num_views):
                total_view_acc[v] += compute_accuracy(outputs['view_logits'][v], labels).item()

            if model.use_global_head:
                loss += criterion(outputs['global_logits'], labels)
                total_global_acc += compute_accuracy(outputs['global_logits'], labels).item()

            total_loss += loss.item()
            num_batches += 1

    return (total_loss / num_batches,
            [acc / num_batches for acc in total_view_acc],
            total_global_acc / num_batches if model.use_global_head else 0)


def main():
    parser = argparse.ArgumentParser(description='MMatch Training')
    parser.add_argument('--data', type=str, default='data_new/Caltech101-20.mat')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--unlabel-ratio', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lambda-u', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 数据加载
    train_loader, test_loader, info = create_dataloaders(
        args.data, unlabel_ratio=args.unlabel_ratio,
        batch_size=args.batch_size, random_seed=args.seed
    )

    # 模型
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

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 分布对齐模块
    da_modules = [DistributionAlignment(info['num_classes']) for _ in range(info['num_views'] + 1)]

    # 优化器
    criterion = nn.CrossEntropyLoss(reduction='none')
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_acc, best_epoch, no_improve = 0.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_x, loss_u, train_acc, mask_ratio = train_epoch(
            model, train_loader, criterion, optimizer, device,
            da_modules, args.lambda_u, args.threshold, args.temperature
        )

        test_loss, view_acc, test_acc = evaluate(model, test_loader, eval_criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} (x={loss_x:.4f}, u={loss_u:.4f}), "
              f"Test Acc={test_acc:.4f}, Mask={mask_ratio:.3f}")

        if test_acc > best_acc + 0.001:
            best_acc, best_epoch, no_improve = test_acc, epoch, 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best model saved!")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest: Epoch {best_epoch}, Accuracy {best_acc:.4f}")


if __name__ == "__main__":
    main()
