"""
工具函数

对应论文架构图:
  compute_accuracy  —— 评估指标计算
  sharpen           —— 模块 (5) Step 5: Sharpening
  generate_pseudo_labels —— 模块 (5): 完整的伪标签生成流程
"""
import torch


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           评估工具函数                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算分类准确率: ACC = mean(argmax(logits) == labels)"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              模块 (5) Step 5: Sharpening (锐化操作)                         ║
# ║                                                                            ║
# ║  降低温度以增强预测的确定性                                                   ║
# ║  公式: sharpened = Normalize( q^(1/T) ),  T = 0.5                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sharpen(probs: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    """锐化操作: q_sharp = Normalize( q^(1/T) )

    对应论文模块 (5) Step 5:
      - T < 1 时增强高概率类别，抑制低概率类别
      - 默认 T=0.5
    """
    sharpened = probs ** (1.0 / T)
    return sharpened / sharpened.sum(dim=1, keepdim=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           模块 (5): Pseudo-Label Generation (伪标签生成)                     ║
# ║                                                                            ║
# ║  完整流程 (对应论文 Step 1 ~ Step 6):                                       ║
# ║    Step 1: Softmax 得到各视图预测概率 p1, p2, ..., pn                       ║
# ║    Step 2: 计算各视图置信度 conf_v = max(p_v)                               ║
# ║    Step 3: 互教学 — 用其他视图的置信度加权平均生成伪标签                        ║
# ║    Step 4: 分布对齐 — aligned = Normalize(q / q_tilde)                     ║
# ║    Step 5: 锐化 — q^(1/T)                                                 ║
# ║    Step 6: 置信度阈值过滤 — mask = (max_prob >= tau)                        ║
# ║                                                                            ║
# ║  输出: y_hat = argmax(sharpened) + confidence mask                         ║
# ║                                                                            ║
# ║  ★★★ 功能开关说明 ★★★                                                      ║
# ║  通过参数可以独立开关以下功能:                                                ║
# ║    - use_mutual_teaching:         互教学 (Step 3)                          ║
# ║    - use_distribution_alignment:  分布对齐 (Step 4)                        ║
# ║    - use_sharpening:              锐化 (Step 5)                            ║
# ║    - memory_bank:                 记忆库平滑 (传None=关闭)                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_pseudo_labels(view_probs_list, da_modules, temperature=0.5, threshold=0.95,
                           memory_bank=None, global_features=None,
                           use_mutual_teaching=True,
                           use_distribution_alignment=True,
                           use_sharpening=True):
    """
    置信度加权互教学伪标签生成

    对应论文模块 (5) 的完整 6 步流程 + Memory Bank 平滑 (公式 6-8)

    Args:
        view_probs_list: 各视图的预测概率列表 (Step 1 已完成，外部传入 softmax 结果)
        da_modules: 分布对齐模块列表 (V+1个，最后一个用于全局)
        temperature: 锐化温度 T (Step 5)
        threshold: 置信度阈值 tau (Step 6)
        memory_bank: Memory Bank 实例 (可选，None 时跳过记忆库平滑)
        global_features: 全局表示 Z = Concat[Z1;...;Zn] (memory_bank 非空时必须提供)

        ★★★ 功能开关参数 ★★★
        use_mutual_teaching: 是否启用互教学 (Step 3)
            - True:  视图v的伪标签 = 其他视图的置信度加权平均 (默认)
            - False: 视图v的伪标签 = 所有视图(含自身)的简单平均
        use_distribution_alignment: 是否启用分布对齐 (Step 4)
            - True:  对预测分布进行 Normalize(q / q_tilde) 对齐 (默认)
            - False: 跳过分布对齐，直接使用原始概率
        use_sharpening: 是否启用锐化 (Step 5)
            - True:  对概率进行 q^(1/T) 锐化 (默认)
            - False: 跳过锐化，直接使用当前概率

    Returns:
        pseudo_targets_list: 伪标签列表 (V+1个，前V个是各视图，最后1个是全局)
        masks_list: 置信度掩码列表 (V+1个)
    """
    num_views = len(view_probs_list)
    pseudo_targets_list = []
    masks_list = []

    # ---- Step 2: 计算各视图置信度 conf_v = max(p_v) ----
    confidences = [probs.max(dim=1)[0] for probs in view_probs_list]

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  内部处理函数: Step 4 + Step 5 + Step 6                     ║
    # ║  根据功能开关决定是否执行分布对齐和锐化                       ║
    # ╚══════════════════════════════════════════════════════════════╝
    def process_pseudo_label(weighted_probs, da_module):
        """Step 4 + Step 5 + Step 6: 分布对齐 -> 锐化 -> 阈值过滤"""
        current_probs = weighted_probs

        # ================================================================
        #  ★ 功能开关: 分布对齐 (Step 4) ★
        #  开启时: aligned = Normalize(q / q_tilde), 消除类别不平衡
        #  关闭时: 跳过, 直接使用原始概率
        # ================================================================
        if use_distribution_alignment and da_module is not None:
            da_module.update(current_probs)
            current_probs = da_module.align(current_probs)

        # ================================================================
        #  ★ 功能开关: 锐化 (Step 5) ★
        #  开启时: q_sharp = Normalize(q^(1/T)), 增强预测确定性
        #  关闭时: 跳过, 直接使用当前概率
        # ================================================================
        if use_sharpening:
            current_probs = sharpen(current_probs, T=temperature)

        # ---- Step 6: 置信度阈值过滤，生成最终伪标签 ----
        max_probs, targets = torch.max(current_probs, dim=1)
        return targets, max_probs >= threshold

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  辅助函数: 置信度加权平均                                    ║
    # ╚══════════════════════════════════════════════════════════════╝
    def weighted_average(probs_list, conf_list):
        """Step 3 辅助函数: 按置信度加权平均多个视图的预测概率"""
        weights = torch.stack(conf_list, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        probs_stack = torch.stack(probs_list, dim=0)
        return (probs_stack * weights.unsqueeze(-1)).sum(dim=0)

    def simple_average(probs_list):
        """简单平均 (互教学关闭时使用): 所有视图概率的均值"""
        return torch.stack(probs_list, dim=0).mean(dim=0)

    # ================================================================
    #  ★ 功能开关: 互教学 (Step 3) ★
    #  开启时: 视图v的伪标签 = 其他视图的置信度加权平均 (排除自身)
    #  关闭时: 视图v的伪标签 = 所有视图(含自身)的简单平均
    # ================================================================
    if use_mutual_teaching:
        # ---- Step 3: 互教学 — 为每个视图生成伪标签 ----
        # 视图 v 的伪标签 = 其他所有视图的置信度加权平均
        for v in range(num_views):
            other_probs = [view_probs_list[i] for i in range(num_views) if i != v]
            other_confs = [confidences[i] for i in range(num_views) if i != v]
            weighted_probs = weighted_average(other_probs, other_confs)
            targets, mask = process_pseudo_label(weighted_probs, da_modules[v] if da_modules else None)
            pseudo_targets_list.append(targets)
            masks_list.append(mask)
    else:
        # ---- 互教学关闭: 所有视图使用相同的简单平均伪标签 ----
        avg_probs = simple_average(view_probs_list)
        for v in range(num_views):
            targets, mask = process_pseudo_label(avg_probs, da_modules[v] if da_modules else None)
            pseudo_targets_list.append(targets)
            masks_list.append(mask)

    # ---- 全局伪标签: 所有视图的置信度加权平均 ----
    if use_mutual_teaching:
        weighted_probs_global = weighted_average(view_probs_list, confidences)
    else:
        weighted_probs_global = simple_average(view_probs_list)

    # ================================================================
    #  ★ 功能开关: Memory Bank 平滑 (论文公式 7-8) ★
    #  开启时 (memory_bank != None): 用记忆库中邻居信息平滑全局概率
    #  关闭时 (memory_bank == None): 跳过记忆库平滑
    # ================================================================
    if memory_bank is not None and global_features is not None:
        weighted_probs_global = memory_bank.refine(global_features, weighted_probs_global)
        memory_bank.update(global_features, weighted_probs_global)

    targets_global, mask_global = process_pseudo_label(
        weighted_probs_global, da_modules[-1] if da_modules else None
    )
    pseudo_targets_list.append(targets_global)
    masks_list.append(mask_global)

    return pseudo_targets_list, masks_list
