"""
分布对齐模块

对应论文架构图:
  模块 (5) Step 4: Distribution Alignment
  q~(t) = m * q~(t-1) + (1-m) * q(t)
  aligned = Normalize(q / q~)
"""
import torch


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║            模块 (5) Step 4: Distribution Alignment (分布对齐)                ║
# ║                                                                            ║
# ║  维护类别概率的指数移动平均 q_tilde                                           ║
# ║  对预测分布进行缩放归一化，消除类别不平衡偏差                                   ║
# ║                                                                            ║
# ║  公式:                                                                     ║
# ║    q_tilde(t) = momentum * q_tilde(t-1) + (1-momentum) * q_batch(t)       ║
# ║    aligned    = Normalize( q / q_tilde )                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DistributionAlignment:
    """
    分布对齐 (Distribution Alignment) 模块

    对应论文模块 (5) 中的 Step 4:
      - 维护类别概率的移动平均 q_tilde
      - 对预测分布进行缩放: q_aligned = Normalize(q / q_tilde)
      - 消除伪标签中的类别分布偏差
    """

    def __init__(self, num_classes: int, momentum: float = 0.999):
        self.num_classes = num_classes
        self.momentum = momentum
        # ---- 初始化为均匀分布 ----
        self.q_tilde = torch.ones(num_classes) / num_classes

    def update(self, probs: torch.Tensor):
        """更新移动平均 q_tilde: q~(t) = m * q~(t-1) + (1-m) * mean(q_batch)"""
        q_batch = probs.mean(dim=0)
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach().cpu()

    def align(self, probs: torch.Tensor) -> torch.Tensor:
        """对预测分布进行对齐: aligned = Normalize(q / q_tilde)"""
        q_tilde = self.q_tilde.to(probs.device)
        q_tilde = torch.clamp(q_tilde, min=1e-6)
        scaled = probs / q_tilde.unsqueeze(0)
        return scaled / scaled.sum(dim=1, keepdim=True)

    def get_distribution(self) -> torch.Tensor:
        """获取当前的移动平均分布"""
        return self.q_tilde.clone()
