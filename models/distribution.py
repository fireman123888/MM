"""
分布对齐模块
"""
import torch


class DistributionAlignment:
    """
    分布对齐 (Distribution Alignment) 模块
    维护类别概率的移动平均 q_tilde，并对预测分布进行缩放和归一化
    """

    def __init__(self, num_classes: int, momentum: float = 0.999):
        self.num_classes = num_classes
        self.momentum = momentum
        self.q_tilde = torch.ones(num_classes) / num_classes

    def update(self, probs: torch.Tensor):
        """更新移动平均 q_tilde"""
        q_batch = probs.mean(dim=0)
        self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach().cpu()

    def align(self, probs: torch.Tensor) -> torch.Tensor:
        """对预测分布进行对齐: q = Normalize(q / q_tilde)"""
        q_tilde = self.q_tilde.to(probs.device)
        q_tilde = torch.clamp(q_tilde, min=1e-6)
        scaled = probs / q_tilde.unsqueeze(0)
        return scaled / scaled.sum(dim=1, keepdim=True)

    def get_distribution(self) -> torch.Tensor:
        """获取当前的移动平均分布"""
        return self.q_tilde.clone()
