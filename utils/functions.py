"""
工具函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LabelConsistencyMatrix(nn.Module):
    """
    标签一致性矩阵 (CSCR)
    量化样本间的语义相似度，用于指导对比学习
    """

    def __init__(self, num_classes: int, soft_label: bool = True):
        """
        Args:
            num_classes: 类别数量
            soft_label: 是否使用软标签（概率）计算一致性
        """
        super().__init__()
        self.num_classes = num_classes
        self.soft_label = soft_label

    def build_matrix(self, labels: torch.Tensor, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        构建标签一致性矩阵

        Args:
            labels: 硬标签 [batch_size] 或 one-hot [batch_size, num_classes]
            probs: 软标签/预测概率 [batch_size, num_classes]（可选）

        Returns:
            一致性矩阵 [batch_size, batch_size]，值在[0,1]之间
        """
        batch_size = labels.size(0)
        device = labels.device

        if self.soft_label and probs is not None:
            # 使用软标签：基于预测概率的余弦相似度
            probs_norm = F.normalize(probs, dim=1)
            consistency = torch.mm(probs_norm, probs_norm.t())
            # 将相似度映射到[0,1]
            consistency = (consistency + 1) / 2
        else:
            # 使用硬标签：相同类别为1，不同类别为0
            if labels.dim() == 1:
                labels_col = labels.view(-1, 1)
                consistency = (labels_col == labels_col.t()).float()
            else:
                # one-hot编码
                consistency = torch.mm(labels.float(), labels.float().t())

        return consistency


class CSCRContrastiveLoss(nn.Module):
    """
    CSCR类别驱动对比恢复损失
    基于标签一致性矩阵构建正负样本对
    """

    def __init__(self, temperature: float = 0.5, use_soft_weight: bool = True):
        """
        Args:
            temperature: 温度参数
            use_soft_weight: 是否使用软权重（一致性分数作为权重）
        """
        super().__init__()
        self.temperature = temperature
        self.use_soft_weight = use_soft_weight
        self.consistency_builder = LabelConsistencyMatrix(num_classes=0, soft_label=True)

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: 特征表示 [batch_size, feature_dim]
            labels: 标签 [batch_size]
            probs: 预测概率 [batch_size, num_classes]（可选，用于软一致性）

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

        # 构建标签一致性矩阵
        if probs is not None:
            consistency = self.consistency_builder.build_matrix(labels, probs)
        else:
            # 硬标签一致性
            labels_col = labels.view(-1, 1)
            consistency = (labels_col == labels_col.t()).float()

        # 排除对角线（自身）
        mask_self = torch.eye(batch_size, device=device)
        consistency = consistency * (1 - mask_self)

        # 检查是否有正样本
        pos_count = consistency.sum(dim=1)
        if pos_count.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 计算对比损失
        exp_sim = torch.exp(similarity) * (1 - mask_self)

        if self.use_soft_weight:
            # 软权重：使用一致性分数作为正样本的权重
            # 负样本权重 = 1 - 一致性分数
            neg_weight = 1 - consistency
            weighted_exp_neg = (exp_sim * neg_weight).sum(dim=1, keepdim=True)
            log_prob = similarity - torch.log(torch.exp(similarity) + weighted_exp_neg + 1e-8)
            # 正样本的加权损失
            weighted_log_prob = (consistency * log_prob).sum(dim=1) / (pos_count + 1e-8)
        else:
            # 硬权重：标准对比损失
            log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            weighted_log_prob = (consistency * log_prob).sum(dim=1) / (pos_count + 1e-8)

        valid_mask = pos_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -weighted_log_prob[valid_mask].mean()
        return loss


class CSCRModule(nn.Module):
    """
    CSCR完整模块：类别驱动对比恢复
    包含标签一致性矩阵构建和对比学习
    """

    def __init__(self, num_classes: int, feature_dim: int, temperature: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature

        self.consistency_builder = LabelConsistencyMatrix(num_classes, soft_label=True)
        self.contrastive_loss = CSCRContrastiveLoss(temperature, use_soft_weight=True)

        # 可选：特征恢复网络（用于缺失视图恢复）
        self.recovery_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def compute_loss(self, features: torch.Tensor, labels: torch.Tensor,
                     probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算CSCR对比损失"""
        return self.contrastive_loss(features, labels, probs)

    def recover_features(self, features: torch.Tensor,
                         consistency_matrix: torch.Tensor) -> torch.Tensor:
        """
        基于一致性矩阵恢复/增强特征

        Args:
            features: 原始特征 [batch_size, feature_dim]
            consistency_matrix: 一致性矩阵 [batch_size, batch_size]

        Returns:
            恢复后的特征 [batch_size, feature_dim]
        """
        # 使用一致性加权的邻居特征进行增强
        weights = F.softmax(consistency_matrix, dim=1)
        aggregated = torch.mm(weights, features)
        recovered = self.recovery_net(aggregated)
        return recovered + features  # 残差连接


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
