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


class MultiViewAttentionFusion(nn.Module):
    """
    多视图注意力融合模块
    使用注意力机制动态融合不同视图的特征
    """

    def __init__(self, encoding_dim: int, num_views: int, hidden_dim: int = None):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.num_views = num_views
        hidden = hidden_dim or encoding_dim

        # 注意力网络：计算每个视图的重要性权重
        self.attention_net = nn.Sequential(
            nn.Linear(encoding_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )

        # 视图间交互的自注意力
        self.query = nn.Linear(encoding_dim, encoding_dim)
        self.key = nn.Linear(encoding_dim, encoding_dim)
        self.value = nn.Linear(encoding_dim, encoding_dim)
        self.scale = encoding_dim ** 0.5

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encodings: 各视图的编码列表，每个形状为 [batch_size, encoding_dim]

        Returns:
            融合后的全局表示 [batch_size, encoding_dim]
        """
        batch_size = encodings[0].size(0)

        # 堆叠所有视图编码: [batch_size, num_views, encoding_dim]
        stacked = torch.stack(encodings, dim=1)

        # 方法1: 简单注意力权重
        attn_scores = self.attention_net(stacked).squeeze(-1)  # [batch_size, num_views]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, num_views]

        # 加权融合
        weighted_sum = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, encoding_dim]

        # 方法2: 自注意力增强
        Q = self.query(stacked)  # [batch_size, num_views, encoding_dim]
        K = self.key(stacked)
        V = self.value(stacked)

        # 计算自注意力
        self_attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch_size, num_views, num_views]
        self_attn = torch.softmax(self_attn, dim=-1)
        self_attn_out = torch.bmm(self_attn, V)  # [batch_size, num_views, encoding_dim]

        # 融合两种方法：加权和 + 自注意力平均
        self_attn_pooled = self_attn_out.mean(dim=1)  # [batch_size, encoding_dim]
        fused = weighted_sum + self_attn_pooled

        return fused, attn_weights


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
    支持注意力融合模式
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
                 global_hidden_dim: int = None,
                 use_attention_fusion: bool = False):
        super().__init__()

        self.num_views = num_views
        self.encoding_dim = encoding_dim
        self.use_global_head = use_global_head
        self.use_attention_fusion = use_attention_fusion

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

        # 注意力融合模块
        if use_attention_fusion:
            self.attention_fusion = MultiViewAttentionFusion(encoding_dim, num_views)
            # 使用注意力融合时，全局表示维度为 encoding_dim
            global_dim = encoding_dim
        else:
            self.attention_fusion = None
            # 拼接模式，全局表示维度为 encoding_dim * num_views
            global_dim = encoding_dim * num_views

        # 全局分类头
        if use_global_head:
            self.global_classifier = GlobalClassificationHead(
                global_dim, num_classes, global_hidden_dim, dropout * 0.6
            )

    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """对所有视图进行编码"""
        return [self.encoders[v](views[v]) for v in range(self.num_views)]

    def classify(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """对所有视图的编码进行分类"""
        return [self.classification_heads[v](encodings[v]) for v in range(self.num_views)]

    def get_global_representation(self, encodings: List[torch.Tensor]):
        """获取全局表示"""
        if self.use_attention_fusion:
            # 使用注意力融合
            fused, attn_weights = self.attention_fusion(encodings)
            return fused, attn_weights
        else:
            # 拼接所有视图编码
            return torch.cat(encodings, dim=1), None

    def forward(self, views: List[torch.Tensor]) -> Dict:
        """前向传播：编码 + 视图分类 + 全局分类"""
        encodings = self.encode(views)
        view_logits = self.classify(encodings)
        global_rep, attn_weights = self.get_global_representation(encodings)

        outputs = {
            'encodings': encodings,
            'view_logits': view_logits,
            'global_representation': global_rep
        }

        if attn_weights is not None:
            outputs['attention_weights'] = attn_weights

        if self.use_global_head:
            outputs['global_logits'] = self.global_classifier(global_rep)

        return outputs
