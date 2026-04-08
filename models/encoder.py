"""
多视图编码器模型

对应论文架构图:
  (2) Multi-View Encoders  —— MLPEncoder
  (3) Dual-Path Prediction —— ClassificationHead + GlobalClassificationHead
  完整系统               —— MultiViewEncoder
"""
import torch
import torch.nn as nn
from typing import List, Dict


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    模块 (2): Multi-View Encoders                           ║
# ║                                                                            ║
# ║  单视图 MLP 编码器: Input -> Linear -> BN -> ReLU -> Dropout -> ... -> Z   ║
# ║  将原始特征映射到 d 维表示空间                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MLPEncoder(nn.Module):
    """MLP编码器，用于单个视图的特征提取

    对应论文:
        MLP Encoder v: Linear -> BN -> ReLU -> Dropout  (×N层)
        输出 Zv (d-dim 编码向量)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.5, use_batch_norm: bool = True):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # ---- 隐藏层: Linear -> BatchNorm -> ReLU -> Dropout ----
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # ---- 输出层: 映射到 encoding_dim 维表示空间 ----
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              模块 (3a): Dual-Path Prediction — 视图分类头                    ║
# ║                                                                            ║
# ║  每个视图的独立分类头: Dropout -> Linear -> logits_v                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ClassificationHead(nn.Module):
    """分类头，用于单个视图的分类任务

    对应论文:
        ClassHead v: Dropout -> Linear
        输出 logits_v (num_classes 维)
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              模块 (3b): Dual-Path Prediction — 全局分类头                    ║
# ║                                                                            ║
# ║  拼接所有视图编码 [Z1; Z2; ...; Zn] 后进行全局分类                            ║
# ║  Linear -> ReLU -> Dropout -> Linear -> global_logits                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GlobalClassificationHead(nn.Module):
    """全局分类头，对拼接后的全局表示进行分类

    对应论文:
        Concatenate [Z1; Z2; ...; Zn]
        Global Classification Head: Linear -> ReLU -> Dropout -> Linear
        输出 global_logits
    """

    def __init__(self, global_dim: int, num_classes: int,
                 hidden_dim: int = None, dropout: float = 0.3):
        super().__init__()

        if hidden_dim is not None:
            layers = [
                nn.Linear(global_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ]
        else:
            layers = [
                nn.Dropout(dropout),
                nn.Linear(global_dim, num_classes)
            ]

        self.classifier = nn.Sequential(*layers)

    def forward(self, global_rep: torch.Tensor) -> torch.Tensor:
        return self.classifier(global_rep)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    完整系统: MultiViewEncoder                               ║
# ║                                                                            ║
# ║  整合模块 (2) + (3)，构成完整的多视图编码-分类系统                             ║
# ║                                                                            ║
# ║  流程:                                                                     ║
# ║    Views -> [Encoder_1,...,Encoder_n] -> [Z1,...,Zn]                        ║
# ║         -> [ClassHead_1,...,ClassHead_n] -> [logits_1,...,logits_n]         ║
# ║         -> Concat [Z1;...;Zn] -> GlobalHead -> global_logits              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MultiViewEncoder(nn.Module):
    """
    多视图编码器系统
    为V个视图分别创建独立的编码器和分类头，支持全局表示和全局分类

    对应论文架构图的模块 (2) + (3):
      - 每个视图有独立的 MLPEncoder + ClassificationHead
      - 全局路径: 拼接所有视图编码 -> GlobalClassificationHead
    """

    def __init__(self,
                 num_views: int,
                 input_dims: List[int],
                 hidden_dims: List[int],
                 encoding_dim: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True,
                 use_global_head: bool = True,
                 global_hidden_dim: int = None):
        super().__init__()

        self.num_views = num_views
        self.encoding_dim = encoding_dim
        self.use_global_head = use_global_head

        # ---- 模块 (2): 为每个视图创建独立的 MLP 编码器 ----
        self.encoders = nn.ModuleList([
            MLPEncoder(input_dims[v], hidden_dims, encoding_dim, dropout, use_batch_norm)
            for v in range(num_views)
        ])

        # ---- 模块 (3a): 为每个视图创建独立的分类头 ----
        self.classification_heads = nn.ModuleList([
            ClassificationHead(encoding_dim, num_classes, dropout * 0.6)
            for v in range(num_views)
        ])

        # ---- 模块 (3b): 全局分类头 ----
        if use_global_head:
            self.global_classifier = GlobalClassificationHead(
                encoding_dim * num_views, num_classes, global_hidden_dim, dropout * 0.6
            )

    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """模块 (2): 对所有视图进行编码，得到 [Z1, Z2, ..., Zn]"""
        return [self.encoders[v](views[v]) for v in range(self.num_views)]

    def classify(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """模块 (3a): 对所有视图的编码进行分类，得到 [logits_1, ..., logits_n]"""
        return [self.classification_heads[v](encodings[v]) for v in range(self.num_views)]

    def get_global_representation(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """模块 (3b): 获取全局表示 Z = Concat[Z1; Z2; ...; Zn]"""
        return torch.cat(encodings, dim=1)

    def forward(self, views: List[torch.Tensor]) -> Dict:
        """前向传播: 模块 (2) 编码 -> 模块 (3a) 视图分类 -> 模块 (3b) 全局分类"""

        # ---- 模块 (2): 编码 ----
        encodings = self.encode(views)

        # ---- 模块 (3a): 各视图分类 ----
        view_logits = self.classify(encodings)

        # ---- 模块 (3b): 全局表示 + 全局分类 ----
        global_rep = self.get_global_representation(encodings)

        outputs = {
            'encodings': encodings,           # [Z1, Z2, ..., Zn]
            'view_logits': view_logits,       # [logits_1, ..., logits_n]
            'global_representation': global_rep  # Concat[Z1;...;Zn]
        }

        if self.use_global_head:
            outputs['global_logits'] = self.global_classifier(global_rep)

        return outputs
