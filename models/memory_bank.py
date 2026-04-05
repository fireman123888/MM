"""
记忆库模块 (Memory Bank)

对应论文公式 (6)-(8):
  Memory Bank M = {q_m, Z_m}，FIFO 先进先出更新

  公式(7) 相似度:
    b_m = exp(Z_c · Z_m) / Σ exp(Z_c · Z_m')

  公式(8) 平滑伪标签:
    Q_c = α·q_c + (1-α)·Σ b_m·q_m

  作用: 利用记忆库中邻居样本的信息平滑伪标签，缓解确认偏差
"""
import torch
import torch.nn.functional as F


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              Memory Bank (记忆库) — 论文公式 (6)-(8)                        ║
# ║                                                                            ║
# ║  存储过去样本的全局表示 Z 和类别概率 q                                        ║
# ║  用 FIFO 队列更新，用邻居相似度加权平滑伪标签                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MemoryBank:
    """
    Memory Bank (记忆库)

    对应论文:
      - 存储: M = {q_m, Z_m}_{m=1}^{M_size}
      - 更新: FIFO 先进先出
      - 平滑: Q_c = α·q_c + (1-α)·Σ b_m·q_m

    Args:
        bank_size: 记忆库容量 (存储多少个样本)
        feature_dim: 全局表示 Z 的维度 (encoding_dim * num_views)
        num_classes: 类别数
        alpha: 平衡因子，控制自身预测 vs 邻居信息的权重 (论文中的 α)
    """

    def __init__(self, bank_size: int, feature_dim: int, num_classes: int,
                 alpha: float = 0.9):
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha

        # ---- FIFO 队列: 存储全局表示 Z_m 和类别概率 q_m ----
        self.features = torch.zeros(bank_size, feature_dim)   # Z_m
        self.probs = torch.zeros(bank_size, num_classes)       # q_m
        self.ptr = 0          # 当前写入位置
        self.num_stored = 0   # 已存储的样本数

    def update(self, features: torch.Tensor, probs: torch.Tensor):
        """
        FIFO 更新记忆库

        将当前 batch 的全局表示和类别概率存入记忆库，
        超出容量时覆盖最早的样本。

        Args:
            features: 全局表示 Z, shape = (batch_size, feature_dim)
            probs: 类别概率 q, shape = (batch_size, num_classes)
        """
        batch_size = features.size(0)
        features = features.detach().cpu()
        probs = probs.detach().cpu()

        # ---- FIFO 写入 ----
        if self.ptr + batch_size <= self.bank_size:
            # 不超出容量，直接写入
            self.features[self.ptr:self.ptr + batch_size] = features
            self.probs[self.ptr:self.ptr + batch_size] = probs
        else:
            # 超出容量，分两段写入 (环形队列)
            remaining = self.bank_size - self.ptr
            self.features[self.ptr:] = features[:remaining]
            self.probs[self.ptr:] = probs[:remaining]
            overflow = batch_size - remaining
            self.features[:overflow] = features[remaining:]
            self.probs[:overflow] = probs[remaining:]

        self.ptr = (self.ptr + batch_size) % self.bank_size
        self.num_stored = min(self.num_stored + batch_size, self.bank_size)

    def refine(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        用记忆库中的邻居信息平滑伪标签 — 论文公式 (7)+(8)

        公式(7): b_m = exp(Z_c · Z_m) / Σ exp(Z_c · Z_m')
        公式(8): Q_c = α·q_c + (1-α)·Σ b_m·q_m

        Args:
            features: 当前样本的全局表示 Z_c, shape = (batch_size, feature_dim)
            probs: 当前样本的类别概率 q_c, shape = (batch_size, num_classes)

        Returns:
            refined_probs: 平滑后的类别概率 Q_c, shape = (batch_size, num_classes)
        """
        # ---- 记忆库为空时直接返回原始概率 ----
        if self.num_stored == 0:
            return probs

        # ---- 取出记忆库中有效的样本 ----
        bank_features = self.features[:self.num_stored].to(features.device)  # (M, D)
        bank_probs = self.probs[:self.num_stored].to(features.device)        # (M, C)

        # ---- 公式(7): 计算相似度 b_m ----
        # L2 归一化后计算 cosine similarity
        features_norm = F.normalize(features, dim=1)          # (B, D)
        bank_features_norm = F.normalize(bank_features, dim=1)  # (M, D)

        # Z_c · Z_m^T -> (B, M)
        similarity = torch.mm(features_norm, bank_features_norm.t())

        # softmax 归一化得到权重 b_m
        b_m = F.softmax(similarity, dim=1)  # (B, M)

        # ---- 公式(8): 加权平均邻居概率 ----
        # Σ b_m · q_m -> (B, C)
        neighbor_probs = torch.mm(b_m, bank_probs)  # (B, C)

        # Q_c = α·q_c + (1-α)·Σ b_m·q_m
        refined_probs = self.alpha * probs + (1 - self.alpha) * neighbor_probs

        return refined_probs

    def is_ready(self) -> bool:
        """记忆库是否有足够的样本可用"""
        return self.num_stored > 0
