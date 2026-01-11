"""
在 Caltech101-7 数据集上训练 MMatch
使用 10% 的数据作为有标签样本
"""
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from data_loader_caltech import create_caltech_dataloaders
from multi_view_encoder import MultiViewEncoder

class DistributionAlignment:
    """分布对齐模块"""
    def __init__(self, num_classes, momentum=0.999, device='cpu'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device
        self.q_tilde = (torch.ones(num_classes) / num_classes).to(device)
    #收取
    def update(self, probs):
        q_batch = probs.mean(dim=0)
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach()
    
    def align(self, probs):
        q_tilde = torch.clamp(self.q_tilde, min=1e-6)
        scaled = probs / q_tilde.unsqueeze(0)
        aligned_probs = scaled / scaled.sum(dim=1, keepdim=True)
        return aligned_probs

    def get_distribution(self):
        return self.q_tilde.clone()


def compute_accuracy(logits, labels):
    """计算准确率"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean() * 100


def sharpen(probs, T):
    """锐化操作"""
    sharpened = probs ** (1 / T)
    return sharpened / sharpened.sum(dim=1, keepdim=True)


def train_epoch_mmatch(model, train_loader, criterion, optimizer, device, epoch,
                       da_module, lambda_u=1.0, threshold=0.95, temperature=0.5):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    total_loss_x = 0
    total_loss_u = 0
    total_global_acc = 0.0
    total_mask_ratio = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # 有标签数据
        labeled_batch = batch['labeled']
        labeled_views = [view.to(device) for view in labeled_batch['views']]
        labels = labeled_batch['label'].to(device)

        # 无标签数据
        unlabeled_batch = batch['unlabeled']
        unlabeled_views = [view.to(device) for view in unlabeled_batch['views']]

        optimizer.zero_grad()

        # 前向传播
        labeled_outputs = model(labeled_views)

        with torch.no_grad():
            unlabeled_outputs = model(unlabeled_views)

        # 计算监督损失 L_x
        loss_x = 0
        for v in range(model.num_views):
            view_loss = criterion(labeled_outputs['view_logits'][v], labels).mean()
            loss_x += view_loss

        if model.use_global_head:
            global_loss = criterion(labeled_outputs['global_logits'], labels).mean()
            loss_x += global_loss

        # 生成伪标签
        loss_u = 0
        mask_ratio = 0
        if model.use_global_head:
            unlabeled_logits = unlabeled_outputs['global_logits']
            unlabeled_probs = F.softmax(unlabeled_logits, dim=1)

            # 分布对齐
            da_module.update(unlabeled_probs)
            aligned_probs = da_module.align(unlabeled_probs)

            # 锐化
            pseudo_labels = sharpen(aligned_probs, T=temperature)
            max_probs, pseudo_targets = torch.max(pseudo_labels, dim=1)

            # 置信度过滤
            mask = max_probs >= threshold
            mask_ratio = mask.float().mean().item()

            # 计算无监督损失 L_u
            if mask.sum() > 0:
                unlabeled_outputs_train = model(unlabeled_views)

                for v in range(model.num_views):
                    view_logits = unlabeled_outputs_train['view_logits'][v]
                    view_loss_u = (criterion(view_logits, pseudo_targets) * mask).mean()
                    loss_u += view_loss_u

                global_logits = unlabeled_outputs_train['global_logits']
                global_loss_u = (criterion(global_logits, pseudo_targets) * mask).mean()
                loss_u += global_loss_u

            total_mask_ratio += mask_ratio

        # 总损失
        loss = loss_x + lambda_u * loss_u

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录统计
        total_loss += loss.item()
        total_loss_x += loss_x.item()
        if isinstance(loss_u, torch.Tensor):
            total_loss_u += loss_u.item()

        if model.use_global_head:
            global_acc = compute_accuracy(labeled_outputs['global_logits'], labels)
            total_global_acc += global_acc.item()

        num_batches += 1

        # 每10个batch打印一次
        if (batch_idx + 1) % 10 == 0:
            print(f'    Iteration {batch_idx + 1}/{len(train_loader)} - '
                  f'Loss: {loss.item():.4f} (L_x: {loss_x.item():.4f}, '
                  f'L_u: {loss_u.item() if isinstance(loss_u, torch.Tensor) else 0:.4f}), '
                  f'Mask Ratio: {mask_ratio:.3f}')

    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_u = total_loss_u / num_batches
    avg_global_acc = total_global_acc / num_batches if model.use_global_head else 0
    avg_mask_ratio = total_mask_ratio / num_batches

    return avg_loss, avg_loss_x, avg_loss_u, avg_global_acc, avg_mask_ratio


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_view_acc = [0.0] * model.num_views
    total_global_acc = 0.0
    num_batches = 0

    # 用于计算 F1 score
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            views = [view.to(device) for view in batch['views']]
            labels = batch['label'].to(device)

            outputs = model(views)

            loss = 0
            for v in range(model.num_views):
                view_loss = criterion(outputs['view_logits'][v], labels).mean()
                loss += view_loss
                acc = compute_accuracy(outputs['view_logits'][v], labels)
                total_view_acc[v] += acc.item()

            if model.use_global_head:
                global_loss = criterion(outputs['global_logits'], labels).mean()
                loss += global_loss
                global_acc = compute_accuracy(outputs['global_logits'], labels)
                total_global_acc += global_acc.item()

                # 收集预测和真实标签用于计算 F1
                preds = torch.argmax(outputs['global_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_view_acc = [acc / num_batches for acc in total_view_acc]
    avg_global_acc = total_global_acc / num_batches if model.use_global_head else 0

    # 计算 F1 score (macro average)
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100

    return avg_loss, avg_view_acc, avg_global_acc, f1_macro


def main():
    """主训练函数"""
    random_seed = random.randint(1, 1000000)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 训练参数
    labeled_ratio = 0.01  # 10% 有标签数据
    test_ratio = 0.2
    batch_size = 8
    mu = 7
    num_epochs = 200
    learning_rate = 0.005
    patience = 40

    # MMatch 参数
    lambda_u = 1.0
    threshold = 0.95
    temperature = 0.5
    da_momentum = 0.999

    print("="*70)
    print("Caltech101-7 MMatch 训练配置")
    print("="*70)
    print(f"有标签数据比例: {labeled_ratio*100:.1f}%")
    print(f"测试集比例: {test_ratio*100:.1f}%")
    print(f"批次大小: {batch_size}")
    print(f"μ (无标签/有标签比例): {mu}")
    print(f"最大训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"早停patience: {patience} epochs")
    print(f"\nMMatch 参数:")
    print(f"  λ_u: {lambda_u}")
    print(f"  置信度阈值: {threshold}")
    print(f"  锐化温度: {temperature}")
    print(f"  DA动量: {da_momentum}")
    print("="*70 + "\n")

    # 加载数据
    print("正在加载数据...")
    random_seed = random.randint(1, 1000000)
    train_loader, test_loader, dataset = create_caltech_dataloaders(
        data_path='Caltech101-7.mat',
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        mu=mu,
        random_seed=random_seed
    )

    # 创建模型
    print("\n正在创建模型...")
    model = MultiViewEncoder(
        num_views=dataset.num_views,
        input_dims=dataset.view_dims,
        hidden_dims=[256, 128],
        encoding_dim=64,
        num_classes=dataset.num_classes,
        use_global_head=True
    ).to(device)

    print(f"\n模型参数:")
    print(f"  视图数量: {model.num_views}")
    print(f"  视图维度: {dataset.view_dims}")
    print(f"  隐藏层维度: [256, 128]")
    print(f"  共享维度: 64")
    print(f"  全局表示维度: {64 * model.num_views}")
    print(f"  类别数量: {dataset.num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 分布对齐模块
    da_module = DistributionAlignment(
        num_classes=dataset.num_classes,
        momentum=da_momentum,
        device=device
    )

    # 训练循环
    print("\n" + "="*70)
    print("开始训练 (MMatch)")
    print("="*70)

    best_acc = 0
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f'  训练 Epoch {epoch}...')

        # 训练
        train_loss, train_loss_x, train_loss_u, train_acc, mask_ratio = train_epoch_mmatch(
            model, train_loader, criterion, optimizer, device, epoch,
            da_module, lambda_u, threshold, temperature
        )

        # 评估
        test_loss, test_view_acc, test_global_acc, test_f1 = evaluate(
            model, test_loader, criterion, device
        )

        # 打印结果
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  训练 - Loss: {train_loss:.4f} (L_x: {train_loss_x:.4f}, L_u: {train_loss_u:.4f}), "
              f"Global Acc: {train_acc:.4f}, Mask Ratio: {mask_ratio:.3f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Global Acc: {test_global_acc:.4f}, F1 Score: {test_f1:.4f}")
        print(f"  视图准确率 (测试):")
        for v, acc in enumerate(test_view_acc):
            print(f"    视图 {v+1}: {acc:.4f}")

        # 保存最佳模型
        if test_global_acc > best_acc:
            best_acc = test_global_acc
            best_f1 = test_f1
            torch.save(model.state_dict(), 'best_model_caltech.pth')
            print(f"  [BEST] Save best model (Global Acc: {best_acc:.4f}, F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停！连续 {patience} 个 epoch 无改进")
            break

    print("\n" + "="*70)
    print("训练完成！")
    print(f"最佳测试准确率: {best_acc:.4f}")
    print(f"最佳测试 F1 Score: {best_f1:.4f}")
    print("="*70)
    return best_acc, best_f1


if __name__ == '__main__':
    total_runs = 5
    acc_list = []
    f1_list = []

    for run_idx in range(total_runs):
        best_acc, best_f1 = main()
        acc_list.append(best_acc)
        f1_list.append(best_f1)
        print(f"===== 第 {run_idx + 1} 次运行结束，最佳准确率：{best_acc:.4f}，最佳 F1：{best_f1:.4f} =====\n")

    # 计算平均值和样本标准差
    avg_acc = statistics.mean(acc_list)
    std_acc = statistics.stdev(acc_list)
    avg_f1 = statistics.mean(f1_list)
    std_f1 = statistics.stdev(f1_list)

    print(f"最终准确率结果：{avg_acc:.4f}（{std_acc:.4f}）")
    print(f"最终 F1 结果：{avg_f1:.4f}（{std_f1:.4f}）")
