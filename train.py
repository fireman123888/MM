"""
训练脚本 - 使用data_loader训练多视图编码器模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from data_loader import create_hw_dataloaders
from multi_view_encoder import MultiViewEncoder


def compute_accuracy(logits, labels):
    """计算分类准确率"""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / len(labels)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_view_acc = [0.0] * model.num_views
    total_global_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # 获取有标签数据
        labeled_batch = batch['labeled']
        labeled_views = [view.to(device) for view in labeled_batch['views']]
        labels = labeled_batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(labeled_views)

        # 计算损失
        loss = 0

        # 每个视图的分类损失
        for v in range(model.num_views):
            view_loss = criterion(outputs['view_logits'][v], labels)
            loss += view_loss

            # 计算准确率
            acc = compute_accuracy(outputs['view_logits'][v], labels)
            total_view_acc[v] += acc.item()

        # 全局分类损失
        if model.use_global_head:
            global_loss = criterion(outputs['global_logits'], labels)
            loss += global_loss

            # 全局准确率
            global_acc = compute_accuracy(outputs['global_logits'], labels)
            total_global_acc += global_acc.item()

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'global_acc': f'{global_acc.item():.4f}' if model.use_global_head else 'N/A'
        })

    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_view_acc = [acc / num_batches for acc in total_view_acc]
    avg_global_acc = total_global_acc / num_batches if model.use_global_head else 0

    return avg_loss, avg_view_acc, avg_global_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_view_acc = [0.0] * model.num_views
    total_global_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            views = [view.to(device) for view in batch['views']]
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(views)

            # 计算损失
            loss = 0
            for v in range(model.num_views):
                view_loss = criterion(outputs['view_logits'][v], labels)
                loss += view_loss

                # 计算准确率
                acc = compute_accuracy(outputs['view_logits'][v], labels)
                total_view_acc[v] += acc.item()

            # 全局分类
            if model.use_global_head:
                global_loss = criterion(outputs['global_logits'], labels)
                loss += global_loss

                global_acc = compute_accuracy(outputs['global_logits'], labels)
                total_global_acc += global_acc.item()

            total_loss += loss.item()
            num_batches += 1

    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_view_acc = [acc / num_batches for acc in total_view_acc]
    avg_global_acc = total_global_acc / num_batches if model.use_global_head else 0

    return avg_loss, avg_view_acc, avg_global_acc


def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 训练参数
    num_labeled = 100
    num_unlabeled = 1900
    batch_size = 32
    mu = 7
    num_epochs = 50
    learning_rate = 0.001

    print("\n" + "="*70)
    print("训练配置")
    print("="*70)
    print(f"有标签样本: {num_labeled}")
    print(f"无标签样本: {num_unlabeled}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print("="*70 + "\n")

    # 创建数据加载器
    print("正在加载数据...")
    train_loader, test_loader, hw_dataset = create_hw_dataloaders(
        num_labeled=num_labeled,
        num_unlabeled=num_unlabeled,
        batch_size=batch_size,
        mu=mu,
        use_mock_data=True,
        random_seed=42
    )

    # 创建模型
    print("\n正在创建模型...")
    model = MultiViewEncoder(
        num_views=hw_dataset.num_views,
        input_dims=list(hw_dataset.view_dims.values()),
        hidden_dims=[256, 128],
        encoding_dim=64,
        num_classes=hw_dataset.num_classes,
        dropout=0.5,
        use_batch_norm=True,
        use_global_head=True,
        global_hidden_dim=128
    ).to(device)

    print(f"\n模型参数:")
    print(f"  视图数量: {hw_dataset.num_views}")
    print(f"  输入维度: {list(hw_dataset.view_dims.values())}")
    print(f"  隐藏层维度: [256, 128]")
    print(f"  编码维度: 64")
    print(f"  全局表示维度: {64 * hw_dataset.num_views}")
    print(f"  类别数量: {hw_dataset.num_classes}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    best_global_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_view_acc, train_global_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 评估
        test_loss, test_view_acc, test_global_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  训练 - Loss: {train_loss:.4f}, Global Acc: {train_global_acc:.4f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Global Acc: {test_global_acc:.4f}")

        # 打印每个视图的准确率
        print(f"  视图准确率 (测试):")
        for v, (view_name, acc) in enumerate(zip(hw_dataset.view_names, test_view_acc)):
            print(f"    {view_name}: {acc:.4f}")

        # 保存最佳模型
        if test_global_acc > best_global_acc:
            best_global_acc = test_global_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_acc': test_global_acc,
                'view_acc': test_view_acc,
            }, 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (Global Acc: {best_global_acc:.4f})")

    # 训练完成
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳模型: Epoch {best_epoch}, Global Acc: {best_global_acc:.4f}")

    # 加载最佳模型并进行最终评估
    print("\n加载最佳模型进行最终评估...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_view_acc, final_global_acc = evaluate(
        model, test_loader, criterion, device
    )

    print("\n最终测试结果:")
    print(f"  全局准确率: {final_global_acc:.4f}")
    print(f"  各视图准确率:")
    for view_name, acc in zip(hw_dataset.view_names, final_view_acc):
        print(f"    {view_name}: {acc:.4f}")

    print("\n模型已保存至: best_model.pth")


if __name__ == "__main__":
    main()
