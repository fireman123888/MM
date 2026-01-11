"""
工具函数
"""
import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算分类准确率"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()


def sharpen(probs: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    """锐化操作：降低温度以增强预测的确定性"""
    sharpened = probs ** (1.0 / T)
    return sharpened / sharpened.sum(dim=1, keepdim=True)


def generate_pseudo_labels(view_probs_list, da_modules, temperature=0.5, threshold=0.95):
    """
    置信度加权互教学伪标签生成

    Args:
        view_probs_list: 各视图的预测概率列表
        da_modules: 分布对齐模块列表 (V+1个)
        temperature: 锐化温度
        threshold: 置信度阈值

    Returns:
        pseudo_targets_list: 伪标签列表 (V+1个)
        masks_list: 置信度掩码列表 (V+1个)
    """
    num_views = len(view_probs_list)
    pseudo_targets_list = []
    masks_list = []

    confidences = [probs.max(dim=1)[0] for probs in view_probs_list]

    def process_pseudo_label(weighted_probs, da_module):
        da_module.update(weighted_probs)
        aligned = da_module.align(weighted_probs)
        sharpened = sharpen(aligned, T=temperature)
        max_probs, targets = torch.max(sharpened, dim=1)
        return targets, max_probs >= threshold

    def weighted_average(probs_list, conf_list):
        weights = torch.stack(conf_list, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        probs_stack = torch.stack(probs_list, dim=0)
        return (probs_stack * weights.unsqueeze(-1)).sum(dim=0)

    # 为每个视图生成伪标签（互教学）
    for v in range(num_views):
        other_probs = [view_probs_list[i] for i in range(num_views) if i != v]
        other_confs = [confidences[i] for i in range(num_views) if i != v]
        weighted_probs = weighted_average(other_probs, other_confs)
        targets, mask = process_pseudo_label(weighted_probs, da_modules[v])
        pseudo_targets_list.append(targets)
        masks_list.append(mask)

    # 全局伪标签
    weighted_probs_global = weighted_average(view_probs_list, confidences)
    targets_global, mask_global = process_pseudo_label(weighted_probs_global, da_modules[-1])
    pseudo_targets_list.append(targets_global)
    masks_list.append(mask_global)

    return pseudo_targets_list, masks_list
