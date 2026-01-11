"""
多视图编码器模型
"""
import torch
import torch.nn as nn
from typing import List, Dict


class MLPEncoder(nn.Module):
    """MLP编码器，用于单个视图的特征提取"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.5, use_batch_norm: bool = True):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ClassificationHead(nn.Module):
    """分类头，用于单个视图的分类任务"""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GlobalClassificationHead(nn.Module):
    """全局分类头，对拼接后的全局表示进行分类"""

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


class MultiViewEncoder(nn.Module):
    """
    多视图编码器系统
    为V个视图分别创建独立的编码器和分类头，支持全局表示和全局分类
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

        # 为每个视图创建独立的编码器
        self.encoders = nn.ModuleList([
            MLPEncoder(input_dims[v], hidden_dims, encoding_dim, dropout, use_batch_norm)
            for v in range(num_views)
        ])

        # 为每个视图创建独立的分类头
        self.classification_heads = nn.ModuleList([
            ClassificationHead(encoding_dim, num_classes, dropout * 0.6)
            for v in range(num_views)
        ])

        # 全局分类头
        if use_global_head:
            self.global_classifier = GlobalClassificationHead(
                encoding_dim * num_views, num_classes, global_hidden_dim, dropout * 0.6
            )

    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """对所有视图进行编码"""
        return [self.encoders[v](views[v]) for v in range(self.num_views)]

    def classify(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """对所有视图的编码进行分类"""
        return [self.classification_heads[v](encodings[v]) for v in range(self.num_views)]

    def get_global_representation(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """获取全局表示 Z (拼接所有视图编码)"""
        return torch.cat(encodings, dim=1)

    def forward(self, views: List[torch.Tensor]) -> Dict:
        """前向传播：编码 + 视图分类 + 全局分类"""
        encodings = self.encode(views)
        view_logits = self.classify(encodings)
        global_rep = self.get_global_representation(encodings)

        outputs = {
            'encodings': encodings,
            'view_logits': view_logits,
            'global_representation': global_rep
        }

        if self.use_global_head:
            outputs['global_logits'] = self.global_classifier(global_rep)

        return outputs
