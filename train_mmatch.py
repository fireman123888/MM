"""
MMatch 训练脚本 - 实现完整的半监督多视图学习
包含：
1. 监督损失 L_x (有标签数据)
2. 无监督损失 L_u (无标签数据，使用伪标签)
3. 分布对齐 (Distribution Alignment)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from data_loader import create_hw_dataloaders
from multi_view_encoder import MultiViewEncoder


class DistributionAlignment:
    """
    分布对齐 (Distribution Alignment) 模块
    维护类别概率的移动平均 q_tilde，并对预测分布进行缩放和归一化
    """
    def __init__(self, num_classes, momentum=0.999):
        """
        Args:
            num_classes: 类别数量
            momentum: 移动平均的动量参数
        """
        self.num_classes = num_classes
        self.momentum = momentum
        # 初始化为均匀分布
        self.q_tilde = torch.ones(num_classes) / num_classes

    def update(self, probs):
        """
        更新移动平均 q_tilde

        Args:
            probs: 当前批次的预测概率分布 (batch_size, num_classes)
        """
        # 计算当前批次的平均分布
        q_batch = probs.mean(dim=0)  # (num_classes,)

        # 移动平均更新: q_tilde = momentum * q_tilde + (1 - momentum) * q_batch
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach()

    def align(self, probs):
        """
        对预测分布进行对齐
        q = Normalize(q / q_tilde)

        Args:
            probs: 预测概率分布 (batch_size, num_classes)

        Returns:
            aligned_probs: 对齐后的概率分布 (batch_size, num_classes)
        """
        # 将 q_tilde 移到相同设备
        q_tilde = self.q_tilde.to(probs.device)

        # 避免除以零
        q_tilde = torch.clamp(q_tilde, min=1e-6)

        # 缩放: q / q_tilde
        scaled = probs / q_tilde.unsqueeze(0)

        # 归一化: Normalize(a)_i = a_i / Σ_j a_j
        aligned_probs = scaled / scaled.sum(dim=1, keepdim=True)

        return aligned_probs

    def get_distribution(self):
        """获取当前的移动平均分布"""
        return self.q_tilde.clone()


def compute_accuracy(logits, labels):
    """计算分类准确率"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()


def sharpen(probs, T=0.5):
    """
    锐化操作：降低温度以增强预测的确定性

    Args:
        probs: 概率分布 (batch_size, num_classes)
        T: 温度参数，越小越锐化

    Returns:
        sharpened_probs: 锐化后的概率分布
    """
    # q^(1/T)
    sharpened = probs ** (1.0 / T)
    # 归一化
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
    return sharpened


def generate_view_pseudo_labels(view_probs_list, da_modules, temperature=0.5, threshold=0.95):
    """
    置信度加权互教学伪标签生成

    Args:
        view_probs_list: 各视图的预测概率列表 [(batch_size, num_classes), ...]
        da_modules: 分布对齐模块列表，长度为 V+1（V个视图 + 1个全局）
        temperature: 锐化温度
        threshold: 置信度阈值

    Returns:
        pseudo_targets_list: 各视图的伪标签列表 + 全局伪标签
        masks_list: 各视图的置信度掩码列表 + 全局掩码
    """
    num_views = len(view_probs_list)
    pseudo_targets_list = []
    masks_list = []

    # 计算各视图的置信度
    confidences = [probs.max(dim=1)[0] for probs in view_probs_list]

    def process_pseudo_label(weighted_probs, da_module):
        """处理加权概率，生成伪标签和掩码"""
        da_module.update(weighted_probs)
        aligned = da_module.align(weighted_probs)
        sharpened = sharpen(aligned, T=temperature)
        max_probs, targets = torch.max(sharpened, dim=1)
        return targets, max_probs >= threshold

    def weighted_average(probs_list, conf_list):
        """计算置信度加权平均"""
        weights = torch.stack(conf_list, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        probs_stack = torch.stack(probs_list, dim=0)
        return (probs_stack * weights.unsqueeze(-1)).sum(dim=0)

    # 为每个视图生成伪标签（排除自己，用其他视图的置信度加权平均）
    for v in range(num_views):
        other_probs = [view_probs_list[i] for i in range(num_views) if i != v]
        other_confs = [confidences[i] for i in range(num_views) if i != v]
        weighted_probs = weighted_average(other_probs, other_confs)
        targets, mask = process_pseudo_label(weighted_probs, da_modules[v])
        pseudo_targets_list.append(targets)
        masks_list.append(mask)

    # 为全局分类器生成伪标签（使用所有视图）
    weighted_probs_global = weighted_average(view_probs_list, confidences)
    targets_global, mask_global = process_pseudo_label(weighted_probs_global, da_modules[-1])
    pseudo_targets_list.append(targets_global)
    masks_list.append(mask_global)

    return pseudo_targets_list, masks_list


def train_epoch_mmatch(model, train_loader, criterion, optimizer, device, epoch,
                       da_modules, lambda_u=1.0, threshold=0.95, temperature=0.5):
    """
    MMatch训练一个epoch（置信度加权互教学版本）

    Args:
        model: 多视图编码器模型
        train_loader: 半监督数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        da_modules: 分布对齐模块列表，长度为 V+1（V个视图 + 1个全局）
        lambda_u: 无监督损失权重
        threshold: 伪标签置信度阈值
        temperature: 锐化温度
    """
    model.train()

    total_loss = 0
    total_loss_x = 0
    total_loss_u = 0
    total_global_acc = 0.0
    total_mask_ratio = 0.0  # 被选中的无标签样本比例
    num_batches = 0

    print(f'  训练 Epoch {epoch}...')

    for batch_idx, batch in enumerate(train_loader):
        # ========== 有标签数据 ==========
        labeled_batch = batch['labeled']
        labeled_views = [view.to(device) for view in labeled_batch['views']]
        labels = labeled_batch['label'].to(device)

        # ========== 无标签数据 ==========
        unlabeled_batch = batch['unlabeled']
        unlabeled_views = [view.to(device) for view in unlabeled_batch['views']]

        # ========== 前向传播 ==========
        optimizer.zero_grad()

        # 有标签数据的预测
        labeled_outputs = model(labeled_views)

        # 无标签数据的预测
        with torch.no_grad():
            unlabeled_outputs = model(unlabeled_views)

        # ========== 计算监督损失 L_x ==========
        loss_x = 0

        # 每个视图的监督损失 L_x^v
        for v in range(model.num_views):
            view_loss = criterion(labeled_outputs['view_logits'][v], labels).mean()
            loss_x += view_loss

        # 全局监督损失 L_x^g
        if model.use_global_head:
            global_loss = criterion(labeled_outputs['global_logits'], labels).mean()
            loss_x += global_loss

        # ========== 生成伪标签 (置信度加权互教学) ==========
        # 获取各视图的预测概率分布
        view_probs_list = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                           for v in range(model.num_views)]

        # 使用置信度加权互教学生成伪标签
        # pseudo_targets_list: [视图1伪标签, ..., 视图V伪标签, 全局伪标签]
        # masks_list: [视图1掩码, ..., 视图V掩码, 全局掩码]
        pseudo_targets_list, masks_list = generate_view_pseudo_labels(
            view_probs_list, da_modules, temperature, threshold
        )

        # 计算平均mask_ratio用于统计
        mask_ratio = sum(mask.float().mean().item() for mask in masks_list) / len(masks_list)

        # ========== 计算无监督损失 L_u ==========
        loss_u = 0

        # 检查是否有任何视图有满足阈值的样本
        any_mask = any(mask.sum().item() > 0 for mask in masks_list)

        if any_mask:
            # 对无标签数据重新进行前向传播（启用梯度）
            unlabeled_outputs_train = model(unlabeled_views)

            # 每个视图的无监督损失（使用各自的伪标签）
            for v in range(model.num_views):
                if masks_list[v].sum() > 0:
                    view_logits = unlabeled_outputs_train['view_logits'][v]
                    # 视图v使用由其他视图生成的伪标签
                    view_loss_u = (criterion(view_logits, pseudo_targets_list[v]) * masks_list[v]).mean()
                    loss_u += view_loss_u

            # 全局无监督损失（使用所有视图加权生成的伪标签）
            if model.use_global_head and masks_list[-1].sum() > 0:
                global_logits = unlabeled_outputs_train['global_logits']
                global_loss_u = (criterion(global_logits, pseudo_targets_list[-1]) * masks_list[-1]).mean()
                loss_u += global_loss_u

        total_mask_ratio += mask_ratio

        # ========== 总损失 ==========
        loss = loss_x + lambda_u * loss_u

        # 反向传播
        loss.backward()
        optimizer.step()

        # ========== 记录统计信息 ==========
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
                  f'Loss: {loss.item():.4f} (L_x: {loss_x.item():.4f}, L_u: {loss_u.item() if isinstance(loss_u, torch.Tensor) else 0:.4f}), '
                  f'Mask Ratio: {mask_ratio:.3f}')

    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_u = total_loss_u / num_batches
    avg_global_acc = total_global_acc / num_batches if model.use_global_head else 0
    avg_mask_ratio = total_mask_ratio / num_batches

    # 打印当前的分布对齐统计
    print(f'    分布对齐统计 (V+1个DA模块):')
    for i, da in enumerate(da_modules[:-1]):
        q_tilde = da.get_distribution()
        print(f'      视图{i+1}: min={q_tilde.min():.4f}, max={q_tilde.max():.4f}, std={q_tilde.std():.4f}')
    q_tilde_global = da_modules[-1].get_distribution()
    print(f'      全局: min={q_tilde_global.min():.4f}, max={q_tilde_global.max():.4f}, std={q_tilde_global.std():.4f}')

    return avg_loss, avg_loss_x, avg_loss_u, avg_global_acc, avg_mask_ratio


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
    print(f"使用设备: {device}\n")

    # 训练参数
    num_labeled = 100
    num_unlabeled = 1900
    batch_size = 32
    mu = 7
    num_epochs = 200
    learning_rate = 0.001
    patience = 20  # 早停
    min_delta = 0.001

    # MMatch 特定参数
    lambda_u = 1.0  # 无监督损失权重
    threshold = 0.95  # 伪标签置信度阈值
    temperature = 0.5  # 锐化温度
    da_momentum = 0.999  # 分布对齐动量

    print("="*70)
    print("MMatch 训练配置")
    print("="*70)
    print(f"有标签样本: {num_labeled}")
    print(f"无标签样本: {num_unlabeled}")
    print(f"批次大小: {batch_size}")
    print(f"μ (无标签/有标签比例): {mu}")
    print(f"最大训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"早停patience: {patience} epochs")
    print(f"\nMMatch 参数:")
    print(f"  λ_u (无监督损失权重): {lambda_u}")
    print(f"  置信度阈值: {threshold}")
    print(f"  锐化温度: {temperature}")
    print(f"  DA动量: {da_momentum}")
    print("="*70 + "\n")

    # 创建数据加载器
    print("正在加载数据...")
    train_loader, test_loader, hw_dataset = create_hw_dataloaders(
        num_labeled=num_labeled,
        num_unlabeled=num_unlabeled,
        batch_size=batch_size,
        mu=mu,
        use_mock_data=False,  # 使用真实数据
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

    # 创建分布对齐模块列表（V个视图 + 1个全局 = V+1个）
    num_views = hw_dataset.num_views
    da_modules = [DistributionAlignment(num_classes=hw_dataset.num_classes, momentum=da_momentum)
                  for _ in range(num_views + 1)]
    print(f"\n创建了 {len(da_modules)} 个分布对齐模块 ({num_views}个视图 + 1个全局)")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(reduction='none')  # 使用 reduction='none' 以支持掩码
    eval_criterion = nn.CrossEntropyLoss()  # 评估时使用标准损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    print("\n" + "="*70)
    print("开始训练 (MMatch)")
    print("="*70)

    best_global_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_loss_x, train_loss_u, train_global_acc, mask_ratio = train_epoch_mmatch(
            model, train_loader, criterion, optimizer, device, epoch,
            da_modules, lambda_u, threshold, temperature
        )

        # 评估
        test_loss, test_view_acc, test_global_acc = evaluate(
            model, test_loader, eval_criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  训练 - Loss: {train_loss:.4f} (L_x: {train_loss_x:.4f}, L_u: {train_loss_u:.4f}), "
              f"Global Acc: {train_global_acc:.4f}, Mask Ratio: {mask_ratio:.3f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Global Acc: {test_global_acc:.4f}")

        # 打印每个视图的准确率
        print(f"  视图准确率 (测试):")
        for v, (view_name, acc) in enumerate(zip(hw_dataset.view_names, test_view_acc)):
            print(f"    {view_name}: {acc:.4f}")

        # 保存最佳模型并检查早停
        if test_global_acc > best_global_acc + min_delta:
            best_global_acc = test_global_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_acc': test_global_acc,
                'view_acc': test_view_acc,
                'da_q_tildes': [da.get_distribution() for da in da_modules],
            }, 'best_model_mmatch.pth')
            print(f"  [BEST] Save best model (Global Acc: {best_global_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # 早停检查
            if epochs_without_improvement >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} epochs. Stopping training.")
                print(f"Best model was at epoch {best_epoch} with Global Acc: {best_global_acc:.4f}")
                break

    # 训练完成
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳模型: Epoch {best_epoch}, Global Acc: {best_global_acc:.4f}")

    # 加载最佳模型并进行最终评估
    print("\n加载最佳模型进行最终评估...")
    checkpoint = torch.load('best_model_mmatch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_view_acc, final_global_acc = evaluate(
        model, test_loader, eval_criterion, device
    )

    print("\n最终测试结果:")
    print(f"  全局准确率: {final_global_acc:.4f}")
    print(f"  各视图准确率:")
    for view_name, acc in zip(hw_dataset.view_names, final_view_acc):
        print(f"    {view_name}: {acc:.4f}")

    print("\n模型已保存至: best_model_mmatch.pth")


if __name__ == "__main__":
    main()
