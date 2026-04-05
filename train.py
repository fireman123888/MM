"""
MMatch 训练脚本

对应论文架构图的完整训练流程:
  模块 (1): 多视图输入 (有标签 + 无标签)
  模块 (2): 多视图编码器
  模块 (3): 双路径预测 (视图分类 + 全局分类)
  模块 (4): 监督损失 L_x
  模块 (5): 伪标签生成 (互教学 + 分布对齐 + 锐化 + 阈值过滤)
  模块 (6): 无监督损失 L_u
  模块 (7): 总损失 L_total = L_x + lambda_u * L_u  +  优化器
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models import MultiViewEncoder, DistributionAlignment, MemoryBank
from data import create_dataloaders
from utils import compute_accuracy, generate_pseudo_labels


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║                    ★★★★★  MMatch 功能开关配置  ★★★★★                                ║
# ║                                                                                    ║
# ║  在这里可以单独开关每一个功能模块，方便进行消融实验                                      ║
# ║  True = 开启该功能 (默认)                                                           ║
# ║  False = 关闭该功能                                                                 ║
# ║                                                                                    ║
# ╠══════════════════════════════════════════════════════════════════════════════════════╣
# ║                                                                                    ║
# ║  use_global_head              全局分类头 (模块 3b)                                   ║
# ║    开: 拼接所有视图编码 -> GlobalHead -> global_logits                                ║
# ║    关: 仅使用各视图独立分类头                                                         ║
# ║                                                                                    ║
# ║  use_mutual_teaching          互教学 (模块 5, Step 3)                                ║
# ║    开: 视图v的伪标签 = 其他视图的置信度加权平均 (排除自身)                                ║
# ║    关: 所有视图使用相同的简单平均伪标签                                                 ║
# ║                                                                                    ║
# ║  use_distribution_alignment   分布对齐 (模块 5, Step 4)                              ║
# ║    开: aligned = Normalize(q / q_tilde), 消除类别不平衡偏差                            ║
# ║    关: 跳过分布对齐, 直接使用原始概率                                                  ║
# ║                                                                                    ║
# ║  use_sharpening               锐化 (模块 5, Step 5)                                 ║
# ║    开: q_sharp = Normalize(q^(1/T)), 增强预测确定性                                   ║
# ║    关: 跳过锐化, 直接使用当前概率                                                     ║
# ║                                                                                    ║
# ║  use_memory_bank              记忆库 (论文公式 6-8)                                  ║
# ║    开: 用历史邻居信息平滑伪标签, 缓解确认偏差                                           ║
# ║    关: 不使用记忆库平滑                                                              ║
# ║                                                                                    ║
# ║  use_early_stopping           早停 (Early Stopping)                                 ║
# ║    开: 连续 patience 个 epoch 无提升时停止训练                                        ║
# ║    关: 始终训练满全部 epochs                                                         ║
# ║                                                                                    ║
# ║  use_cosine_scheduler         余弦退火学习率调度 (CosineAnnealingLR)                  ║
# ║    开: 学习率按余弦曲线从 lr 递减到 0                                                 ║
# ║    关: 学习率保持恒定不变                                                             ║
# ║                                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

FEATURE_FLAGS = {
    'use_global_head':              True,   # ★ 全局分类头 (模块 3b)
    'use_mutual_teaching':          True,   # ★ 互教学 (模块 5, Step 3)
    'use_distribution_alignment':   True,   # ★ 分布对齐 (模块 5, Step 4)
    'use_sharpening':               True,   # ★ 锐化 (模块 5, Step 5)
    'use_memory_bank':              True,   # ★ 记忆库 (论文公式 6-8)
    'use_early_stopping':           True,   # ★ 早停 (Early Stopping)
    'use_cosine_scheduler':         True,   # ★ 余弦退火学习率调度
}


def train_epoch(model, train_loader, criterion, optimizer, device,
                da_modules, memory_bank=None, lambda_u=1.0, threshold=0.95, temperature=0.5,
                feature_flags=None):
    """训练一个 epoch，包含模块 (4)~(7) 的完整流程 + Memory Bank"""
    if feature_flags is None:
        feature_flags = FEATURE_FLAGS

    model.train()
    total_loss, total_loss_x, total_loss_u = 0, 0, 0
    total_global_acc, total_mask_ratio = 0.0, 0.0
    num_batches = 0

    for batch in train_loader:

        # ================================================================
        #  模块 (1): Multi-View Input — 加载有标签和无标签的多视图数据
        # ================================================================
        labeled_views = [v.to(device) for v in batch['labeled']['views']]
        labels = batch['labeled']['label'].to(device)
        unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

        optimizer.zero_grad()

        # ================================================================
        #  模块 (2)+(3): 有标签数据前向传播 — 编码 + 双路径分类
        # ================================================================
        labeled_outputs = model(labeled_views)

        # ================================================================
        #  模块 (4): Supervised Loss (监督损失)
        #  L_x = Sum_v CE(logits_v, y) + CE(global_logits, y)
        # ================================================================
        loss_x = sum(criterion(labeled_outputs['view_logits'][v], labels).mean()
                     for v in range(model.num_views))

        # ================================================================
        #  ★ 功能开关: 全局分类头 (use_global_head) ★
        #  开启时: 监督损失额外加上全局分类头的交叉熵损失
        #  关闭时: 仅使用各视图独立分类头的损失
        # ================================================================
        if model.use_global_head:
            loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

        # ================================================================
        #  模块 (5): Pseudo-Label Generation (伪标签生成)
        #  Step 1: Softmax 得到概率
        #  Step 2~6: 互教学 -> 分布对齐 -> 锐化 -> 阈值过滤
        #
        #  ★ 功能开关在 generate_pseudo_labels 内部控制 ★
        #  - use_mutual_teaching:        互教学
        #  - use_distribution_alignment: 分布对齐
        #  - use_sharpening:             锐化
        #  - memory_bank:                记忆库 (传None=关闭)
        # ================================================================
        with torch.no_grad():
            unlabeled_outputs = model(unlabeled_views)
            # ---- Step 1: Softmax 得到各视图预测概率 ----
            view_probs_list = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                               for v in range(model.num_views)]
            # ---- 获取全局表示 (Memory Bank 需要) ----
            global_features = unlabeled_outputs['global_representation']

        # ---- Step 2~6: 生成伪标签和置信度掩码 (含 Memory Bank 平滑) ----
        pseudo_targets_list, masks_list = generate_pseudo_labels(
            view_probs_list,
            da_modules if feature_flags['use_distribution_alignment'] else None,
            temperature, threshold,
            memory_bank=memory_bank if feature_flags['use_memory_bank'] else None,
            global_features=global_features if feature_flags['use_memory_bank'] else None,
            use_mutual_teaching=feature_flags['use_mutual_teaching'],
            use_distribution_alignment=feature_flags['use_distribution_alignment'],
            use_sharpening=feature_flags['use_sharpening'],
        )

        mask_ratio = sum(m.float().mean().item() for m in masks_list) / len(masks_list)

        # ================================================================
        #  模块 (6): Unsupervised Loss (无监督损失)
        #  L_u = Sum_v [CE(logits_v, y_hat_v) * mask_v]
        #       + CE(global_logits, y_hat_g) * mask_g
        # ================================================================
        loss_u = 0
        if any(m.sum().item() > 0 for m in masks_list):
            unlabeled_outputs_train = model(unlabeled_views)
            # ---- 各视图的无监督损失 ----
            for v in range(model.num_views):
                if masks_list[v].sum() > 0:
                    loss_u += (criterion(unlabeled_outputs_train['view_logits'][v],
                                         pseudo_targets_list[v]) * masks_list[v]).mean()

            # ================================================================
            #  ★ 功能开关: 全局分类头的无监督损失 (use_global_head) ★
            #  开启时: 无监督损失额外加上全局路径的伪标签损失
            #  关闭时: 仅使用各视图路径的伪标签损失
            # ================================================================
            if model.use_global_head and masks_list[-1].sum() > 0:
                loss_u += (criterion(unlabeled_outputs_train['global_logits'],
                                     pseudo_targets_list[-1]) * masks_list[-1]).mean()

        # ================================================================
        #  模块 (7): Total Loss & Optimization
        #  L_total = L_x + lambda_u * L_u
        #  Optimizer: Adam | Scheduler: CosineAnnealing
        # ================================================================
        loss = loss_x + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        # ---- 统计信息 ----
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      推理/评估 (Inference)                                  ║
# ║                                                                            ║
# ║  对应论文底部:                                                               ║
# ║  Inference: Multi-View Input -> Encoders -> Global Head -> Prediction      ║
# ║  (不使用伪标签，仅前向传播)                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def evaluate(model, test_loader, criterion, device):
    """评估模型 — 纯推理，不使用伪标签"""
    model.eval()
    total_loss, total_global_acc = 0, 0.0
    total_view_acc = [0.0] * model.num_views
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            views = [v.to(device) for v in batch['views']]
            labels = batch['label'].to(device)

            # ---- 模块 (2)+(3): 编码 + 分类 ----
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         主函数: 完整训练流程                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

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

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  ★★★ 命令行功能开关参数 ★★★                                              ║
    # ║                                                                        ║
    # ║  使用 --no-xxx 关闭对应功能, 例如:                                       ║
    # ║    python train.py --no-memory-bank          关闭记忆库                  ║
    # ║    python train.py --no-mutual-teaching      关闭互教学                  ║
    # ║    python train.py --no-distribution-alignment 关闭分布对齐              ║
    # ║    python train.py --no-sharpening           关闭锐化                   ║
    # ║    python train.py --no-global-head          关闭全局分类头              ║
    # ║    python train.py --no-early-stopping       关闭早停                   ║
    # ║    python train.py --no-cosine-scheduler     关闭余弦退火               ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    parser.add_argument('--no-global-head', action='store_true',
                        help='★ 关闭全局分类头 (模块 3b)')
    parser.add_argument('--no-mutual-teaching', action='store_true',
                        help='★ 关闭互教学 (模块 5, Step 3)')
    parser.add_argument('--no-distribution-alignment', action='store_true',
                        help='★ 关闭分布对齐 (模块 5, Step 4)')
    parser.add_argument('--no-sharpening', action='store_true',
                        help='★ 关闭锐化 (模块 5, Step 5)')
    parser.add_argument('--no-memory-bank', action='store_true',
                        help='★ 关闭记忆库 (论文公式 6-8)')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='★ 关闭早停')
    parser.add_argument('--no-cosine-scheduler', action='store_true',
                        help='★ 关闭余弦退火学习率调度')
    args = parser.parse_args()

    # ================================================================
    #  ★ 解析功能开关 ★
    # ================================================================
    feature_flags = {
        'use_global_head':              not args.no_global_head,
        'use_mutual_teaching':          not args.no_mutual_teaching,
        'use_distribution_alignment':   not args.no_distribution_alignment,
        'use_sharpening':               not args.no_sharpening,
        'use_memory_bank':              not args.no_memory_bank,
        'use_early_stopping':           not args.no_early_stopping,
        'use_cosine_scheduler':         not args.no_cosine_scheduler,
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ================================================================
    #  ★ 打印功能开关状态 ★
    # ================================================================
    print("=" * 60)
    print("  ★★★ 功能开关状态 ★★★")
    print("=" * 60)
    for flag, enabled in feature_flags.items():
        status = "✓ 开启" if enabled else "✗ 关闭"
        print(f"  {flag:<35s} {status}")
    print("=" * 60)
    print()

    # ================================================================
    #  模块 (1): 数据加载 — 多视图输入 (有标签 X_L + 无标签 X_U + 测试集)
    # ================================================================
    train_loader, test_loader, info = create_dataloaders(
        args.data, unlabel_ratio=args.unlabel_ratio,
        batch_size=args.batch_size, random_seed=args.seed
    )

    # ================================================================
    #  模块 (2)+(3): 创建多视图编码器模型
    #
    #  ★ 功能开关: use_global_head ★
    #  控制是否创建全局分类头
    # ================================================================
    model = MultiViewEncoder(
        num_views=info['num_views'],
        input_dims=info['view_dims'],
        hidden_dims=[256, 128],
        encoding_dim=64,
        num_classes=info['num_classes'],
        dropout=0.5,
        use_global_head=feature_flags['use_global_head'],
        global_hidden_dim=128
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ================================================================
    #  ★ 功能开关: 分布对齐模块 (use_distribution_alignment) ★
    #  开启时: 创建 V+1 个分布对齐模块
    #  关闭时: 设为 None, 跳过分布对齐步骤
    # ================================================================
    if feature_flags['use_distribution_alignment']:
        da_modules = [DistributionAlignment(info['num_classes']) for _ in range(info['num_views'] + 1)]
        print("  分布对齐模块: ✓ 已创建")
    else:
        da_modules = None
        print("  分布对齐模块: ✗ 已关闭")

    # ================================================================
    #  ★ 功能开关: Memory Bank 记忆库 (use_memory_bank) ★
    #  开启时: 创建记忆库, 存储历史全局表示和类别概率
    #  关闭时: 设为 None, 跳过记忆库平滑
    # ================================================================
    if feature_flags['use_memory_bank']:
        memory_bank = MemoryBank(
            bank_size=256,
            feature_dim=64 * info['num_views'],
            num_classes=info['num_classes'],
            alpha=0.9
        )
        print("  记忆库: ✓ 已创建 (bank_size=256)")
    else:
        memory_bank = None
        print("  记忆库: ✗ 已关闭")

    # ================================================================
    #  模块 (7): 优化器配置
    # ================================================================
    criterion = nn.CrossEntropyLoss(reduction='none')
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ================================================================
    #  ★ 功能开关: 余弦退火学习率调度 (use_cosine_scheduler) ★
    #  开启时: 学习率按余弦曲线从 lr 递减到 0
    #  关闭时: 学习率保持恒定不变
    # ================================================================
    if feature_flags['use_cosine_scheduler']:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print("  余弦退火调度: ✓ 已创建")
    else:
        scheduler = None
        print("  余弦退火调度: ✗ 已关闭")

    print()

    # ================================================================
    #  训练循环: 每个 epoch 执行 模块 (4)~(7) 的完整流程
    # ================================================================
    best_acc, best_epoch, no_improve = 0.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_x, loss_u, train_acc, mask_ratio = train_epoch(
            model, train_loader, criterion, optimizer, device,
            da_modules, memory_bank, args.lambda_u, args.threshold, args.temperature,
            feature_flags=feature_flags
        )

        test_loss, view_acc, test_acc = evaluate(model, test_loader, eval_criterion, device)

        # ================================================================
        #  ★ 功能开关: 余弦退火学习率调度 (use_cosine_scheduler) ★
        # ================================================================
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} (x={loss_x:.4f}, u={loss_u:.4f}), "
              f"Test Acc={test_acc:.4f}, Mask={mask_ratio:.3f}")

        if test_acc > best_acc + 0.001:
            best_acc, best_epoch, no_improve = test_acc, epoch, 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best model saved!")
        else:
            no_improve += 1

            # ================================================================
            #  ★ 功能开关: 早停 (use_early_stopping) ★
            #  开启时: 连续 patience 个 epoch 无提升时停止训练
            #  关闭时: 始终训练满全部 epochs
            # ================================================================
            if feature_flags['use_early_stopping'] and no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest: Epoch {best_epoch}, Accuracy {best_acc:.4f}")


if __name__ == "__main__":
    main()
