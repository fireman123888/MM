"""
工具函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDrivenContrastiveLoss(nn.Module):
    """
    标签驱动对比损失
    同一类别的样本在表示空间中接近，不同类别的样本远离
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征表示 [batch_size, feature_dim]
            labels: 标签 [batch_size]

        Returns:
            对比损失
        """
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature

        # 构建标签掩码：同类为正样本
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(device)

        # 排除对角线（自身）
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos * (1 - mask_self)

        # 检查是否有正样本对
        pos_count = mask_pos.sum(dim=1)
        if pos_count.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 计算对比损失 (InfoNCE style)
        exp_sim = torch.exp(similarity) * (1 - mask_self)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 只计算有正样本的损失
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (pos_count + 1e-8)
        valid_mask = pos_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -mean_log_prob[valid_mask].mean()
        return loss


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
