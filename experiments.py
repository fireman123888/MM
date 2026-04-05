"""
MMatch 批量实验脚本

对应论文架构图的完整流程，对多个数据集 × 多个无标签比例 × 多个随机种子进行实验:
  模块 (1)~(7) 的完整训练 + 推理评估
  输出: ACC 和 F1 的 mean(std)
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from models import MultiViewEncoder, DistributionAlignment, MemoryBank
from data import create_dataloaders
from utils import generate_pseudo_labels


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           实验配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SEEDS = [51, 52, 53, 54, 55]
UNLABEL_RATIOS = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
DATASETS_NEW = [
    ('data_new', 'Caltech101-20.mat'),
    ('data_new', 'Caltech-5V.mat'),
    ('data_new', 'COIL20.mat'),
    ('data_new', 'cub_googlenet_doc2vec_c10.mat'),
    ('data_new', 'Out_Scene.mat'),
    ('data_new', 'WebKB.mat'),
]
DATASETS_2 = [
    ('data2', '3sources.mat'),
    ('data2', 'Caltech101-20.mat'),
    ('data2', 'GRAZ02.mat'),
    ('data2', 'handwritten.mat'),
    ('data2', 'MSRCv1.mat'),
    ('data2', 'Scene15.mat'),
]
ALL_DATASETS = DATASETS_NEW + DATASETS_2

# ---- 模块 (7) 超参数 ----
BATCH_SIZE = 32
MU = 7
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20
LAMBDA_U = 1.0
THRESHOLD = 0.95
TEMPERATURE = 0.5


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


def train_and_evaluate(mat_path, unlabel_ratio, seed, feature_flags=None):
    """单次实验: 训练 + 评估，返回最佳 ACC 和 F1"""
    if feature_flags is None:
        feature_flags = FEATURE_FLAGS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ================================================================
    #  模块 (1): 数据加载与半监督划分
    # ================================================================
    train_loader, test_loader, info = create_dataloaders(
        mat_path, unlabel_ratio=unlabel_ratio,
        batch_size=BATCH_SIZE, mu=MU, random_seed=seed
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

    # ================================================================
    #  ★ 功能开关: 分布对齐模块 (use_distribution_alignment) ★
    #  开启时: 创建 V+1 个分布对齐模块
    #  关闭时: 设为 None
    # ================================================================
    if feature_flags['use_distribution_alignment']:
        da_modules = [DistributionAlignment(info['num_classes']) for _ in range(info['num_views'] + 1)]
    else:
        da_modules = None

    # ================================================================
    #  ★ 功能开关: Memory Bank 记忆库 (use_memory_bank) ★
    #  开启时: 创建记忆库
    #  关闭时: 设为 None
    # ================================================================
    if feature_flags['use_memory_bank']:
        memory_bank = MemoryBank(
            bank_size=256,
            feature_dim=64 * info['num_views'],
            num_classes=info['num_classes'],
            alpha=0.9
        )
    else:
        memory_bank = None

    # ---- 模块 (7): 优化器 ----
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # ================================================================
    #  ★ 功能开关: 余弦退火学习率调度 (use_cosine_scheduler) ★
    #  开启时: 学习率按余弦曲线从 lr 递减到 0
    #  关闭时: 学习率保持恒定不变
    # ================================================================
    if feature_flags['use_cosine_scheduler']:
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    else:
        scheduler = None

    best_acc, no_improve = 0.0, 0
    best_preds, best_labels = None, None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_loader:

            # ============================================================
            #  模块 (1): 加载有标签/无标签多视图数据
            # ============================================================
            labeled_views = [v.to(device) for v in batch['labeled']['views']]
            labels = batch['labeled']['label'].to(device)
            unlabeled_views = [v.to(device) for v in batch['unlabeled']['views']]

            optimizer.zero_grad()

            # ============================================================
            #  模块 (2)+(3): 有标签数据前向传播
            # ============================================================
            labeled_outputs = model(labeled_views)

            # ============================================================
            #  模块 (4): 监督损失 L_x
            # ============================================================
            loss_x = sum(criterion(labeled_outputs['view_logits'][v], labels).mean()
                         for v in range(model.num_views))

            # ============================================================
            #  ★ 功能开关: 全局分类头 (use_global_head) ★
            #  开启时: 监督损失额外加上全局分类头的交叉熵损失
            # ============================================================
            if model.use_global_head:
                loss_x += criterion(labeled_outputs['global_logits'], labels).mean()

            # ============================================================
            #  模块 (5): 伪标签生成
            #
            #  ★ 内部功能开关 ★
            #  - use_mutual_teaching:        互教学
            #  - use_distribution_alignment: 分布对齐
            #  - use_sharpening:             锐化
            #  - memory_bank:                记忆库
            # ============================================================
            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_views)
                view_probs = [F.softmax(unlabeled_outputs['view_logits'][v], dim=1)
                              for v in range(model.num_views)]
                global_features = unlabeled_outputs['global_representation']

            pseudo_targets, masks = generate_pseudo_labels(
                view_probs,
                da_modules if feature_flags['use_distribution_alignment'] else None,
                TEMPERATURE, THRESHOLD,
                memory_bank=memory_bank if feature_flags['use_memory_bank'] else None,
                global_features=global_features if feature_flags['use_memory_bank'] else None,
                use_mutual_teaching=feature_flags['use_mutual_teaching'],
                use_distribution_alignment=feature_flags['use_distribution_alignment'],
                use_sharpening=feature_flags['use_sharpening'],
            )

            # ============================================================
            #  模块 (6): 无监督损失 L_u
            # ============================================================
            loss_u = 0
            if any(m.sum().item() > 0 for m in masks):
                unlabeled_outputs_train = model(unlabeled_views)
                for v in range(model.num_views):
                    if masks[v].sum() > 0:
                        loss_u += (criterion(unlabeled_outputs_train['view_logits'][v],
                                             pseudo_targets[v]) * masks[v]).mean()

                # ============================================================
                #  ★ 功能开关: 全局分类头的无监督损失 (use_global_head) ★
                # ============================================================
                if model.use_global_head and masks[-1].sum() > 0:
                    loss_u += (criterion(unlabeled_outputs_train['global_logits'],
                                         pseudo_targets[-1]) * masks[-1]).mean()

            # ============================================================
            #  模块 (7): 总损失 + 反向传播
            # ============================================================
            loss = loss_x + LAMBDA_U * loss_u
            loss.backward()
            optimizer.step()

        # ================================================================
        #  ★ 功能开关: 余弦退火学习率调度 (use_cosine_scheduler) ★
        # ================================================================
        if scheduler is not None:
            scheduler.step()

        # ================================================================
        #  推理评估: Multi-View Input -> Encoders -> Global Head -> Pred
        # ================================================================
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                views = [v.to(device) for v in batch['views']]
                outputs = model(views)

                # ============================================================
                #  ★ 功能开关: 评估时的预测方式 (use_global_head) ★
                #  开启时: 使用全局分类头的输出作为最终预测
                #  关闭时: 使用所有视图 logits 的平均作为最终预测
                # ============================================================
                if model.use_global_head:
                    preds = outputs['global_logits'].argmax(dim=1).cpu().numpy()
                else:
                    avg_logits = torch.stack(outputs['view_logits'], dim=0).mean(dim=0)
                    preds = avg_logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())

        acc = accuracy_score(all_labels, all_preds)

        if acc > best_acc + 0.001:
            best_acc, no_improve = acc, 0
            best_preds = list(all_preds)
            best_labels = list(all_labels)
        else:
            no_improve += 1

            # ================================================================
            #  ★ 功能开关: 早停 (use_early_stopping) ★
            #  开启时: 连续 patience 个 epoch 无提升时停止训练
            #  关闭时: 始终训练满全部 epochs
            # ================================================================
            if feature_flags['use_early_stopping'] and no_improve >= PATIENCE:
                break

    best_f1 = f1_score(best_labels, best_preds, average='macro')
    return best_acc, best_f1


def fmt(values):
    """格式化为 mean(std)，小数点后四位"""
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.4f}({std:.4f})"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    主函数: 批量实验 + 汇总输出                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 90)
    print("MMatch 批量实验")
    print(f"Seeds: {SEEDS}  |  Unlabel Ratios: {UNLABEL_RATIOS}")
    print(f"Epochs: {NUM_EPOCHS}  |  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}")

    # ================================================================
    #  ★ 打印功能开关状态 ★
    # ================================================================
    print()
    print("  ★★★ 功能开关状态 ★★★")
    print("-" * 60)
    for flag, enabled in FEATURE_FLAGS.items():
        status = "✓ 开启" if enabled else "✗ 关闭"
        print(f"  {flag:<35s} {status}")
    print("-" * 60)

    print("=" * 90)

    results = {}

    for data_dir, dataset in ALL_DATASETS:
        mat_path = os.path.join(data_dir, dataset)
        if not os.path.exists(mat_path):
            print(f"\n[跳过] {data_dir}/{dataset} (文件不存在)")
            continue

        display_name = f"{data_dir}/{dataset}"
        print(f"\n{'='*90}")
        print(f"数据集: {display_name}")
        print(f"{'='*90}")

        header = f"{'Unlabel Ratio':<15} {'ACC':<20} {'F1':<20}"
        print(header)
        print("-" * 55)

        results[display_name] = {}

        for unlabel_ratio in UNLABEL_RATIOS:
            accs, f1s = [], []
            for seed in SEEDS:
                try:
                    acc, f1 = train_and_evaluate(mat_path, unlabel_ratio, seed, FEATURE_FLAGS)
                    accs.append(acc)
                    f1s.append(f1)
                    sys.stdout.write(f"\r  ratio={unlabel_ratio}, seed={seed} done (acc={acc:.4f}, f1={f1:.4f})")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\n  [错误] ratio={unlabel_ratio}, seed={seed}: {e}")
                    continue

            if len(accs) > 0:
                acc_str = fmt(accs)
                f1_str = fmt(f1s)
                results[display_name][unlabel_ratio] = {
                    'acc': (np.mean(accs), np.std(accs)),
                    'f1': (np.mean(f1s), np.std(f1s))
                }
                print(f"\r  {unlabel_ratio:<15} {acc_str:<20} {f1_str:<20}")

    # ================================================================
    #  汇总结果表格
    # ================================================================
    print(f"\n\n{'='*90}")
    print("汇总结果")
    print(f"{'='*90}")

    for display_name, ratios in results.items():
        print(f"\n{display_name}")
        print(f"  {'Unlabel Ratio':<15} {'ACC':<20} {'F1':<20}")
        print(f"  {'-'*55}")
        for ratio in UNLABEL_RATIOS:
            if ratio in ratios:
                acc_mean, acc_std = ratios[ratio]['acc']
                f1_mean, f1_std = ratios[ratio]['f1']
                print(f"  {ratio:<15} {acc_mean:.4f}({acc_std:.4f})    {f1_mean:.4f}({f1_std:.4f})")


if __name__ == "__main__":
    main()
