import torch
import torch.nn as nn
from typing import List, Dict


class MLPEncoder(nn.Module):
    """
    MLP编码器，用于单个视图的特征提取
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.5, use_batch_norm: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如 [512, 256]
            output_dim: 输出特征维度（编码后的维度）
            dropout: Dropout概率
            use_batch_norm: 是否使用BatchNorm
        """
        super(MLPEncoder, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch_size, input_dim)
        Returns:
            编码后的特征 (batch_size, output_dim)
        """
        return self.encoder(x)


class ClassificationHead(nn.Module):
    """
    分类头，用于单个视图的分类任务
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        """
        Args:
            input_dim: 输入特征维度（编码器输出维度）
            num_classes: 类别数量
            dropout: Dropout概率
        """
        super(ClassificationHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码后的特征 (batch_size, input_dim)
        Returns:
            分类logits (batch_size, num_classes)
        """
        return self.classifier(x)


class GlobalRepresentation(nn.Module):
    """
    全局表示模块
    将所有视图的特征拼接(concatenate)形成全局表示 Z
    """
    def __init__(self, encoding_dim: int, num_views: int):
        """
        Args:
            encoding_dim: 每个视图的编码维度
            num_views: 视图数量 V
        """
        super(GlobalRepresentation, self).__init__()
        self.encoding_dim = encoding_dim
        self.num_views = num_views
        self.global_dim = encoding_dim * num_views  # Z的维度 = V * encoding_dim

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        将所有视图的编码拼接成全局表示

        Args:
            encodings: 列表，包含V个视图的编码特征
                      每个元素为 (batch_size, encoding_dim)
        Returns:
            global_rep: 全局表示 Z (batch_size, V * encoding_dim)
        """
        # 在特征维度上拼接所有视图
        global_rep = torch.cat(encodings, dim=1)
        return global_rep


class GlobalClassificationHead(nn.Module):
    """
    全局分类头 H_Φ
    对拼接后的全局表示进行分类
    """
    def __init__(self, global_dim: int, num_classes: int,
                 hidden_dim: int = None, dropout: float = 0.3):
        """
        Args:
            global_dim: 全局表示的维度（V * encoding_dim）
            num_classes: 类别数量
            hidden_dim: 可选的隐藏层维度，如果提供则添加一个隐藏层
            dropout: Dropout概率
        """
        super(GlobalClassificationHead, self).__init__()

        layers = []
        if hidden_dim is not None:
            # 带隐藏层的全局分类头
            layers.extend([
                nn.Linear(global_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            # 简单的线性分类头
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(global_dim, num_classes)
            ])

        self.classifier = nn.Sequential(*layers)

    def forward(self, global_rep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_rep: 全局表示 Z (batch_size, global_dim)
        Returns:
            logits: 分类logits (batch_size, num_classes)
        """
        return self.classifier(global_rep)


class MultiViewEncoder(nn.Module):
    """
    多视图编码器系统
    为V个视图分别创建独立的编码器(f^v)和分类头(h^v)
    同时支持全局表示Z和全局分类头H_Φ
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
        """
        Args:
            num_views: 视图数量 V
            input_dims: 每个视图的输入维度列表，长度为V
            hidden_dims: MLP隐藏层维度列表，例如 [512, 256]
            encoding_dim: 编码后的特征维度（所有视图统一）
            num_classes: 分类任务的类别数量
            dropout: Dropout概率
            use_batch_norm: 是否使用BatchNorm
            use_global_head: 是否使用全局分类头
            global_hidden_dim: 全局分类头的隐藏层维度（可选）
        """
        super(MultiViewEncoder, self).__init__()

        self.num_views = num_views
        self.encoding_dim = encoding_dim
        self.use_global_head = use_global_head

        # 为每个视图创建独立的编码器 f^v
        self.encoders = nn.ModuleList([
            MLPEncoder(
                input_dim=input_dims[v],
                hidden_dims=hidden_dims,
                output_dim=encoding_dim,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            for v in range(num_views)
        ])

        # 为每个视图创建独立的分类头 h^v
        self.classification_heads = nn.ModuleList([
            ClassificationHead(
                input_dim=encoding_dim,
                num_classes=num_classes,
                dropout=dropout * 0.6  # 分类头使用稍小的dropout
            )
            for v in range(num_views)
        ])

        # 全局表示模块
        self.global_representation = GlobalRepresentation(
            encoding_dim=encoding_dim,
            num_views=num_views
        )

        # 全局分类头 H_Φ
        if use_global_head:
            self.global_classifier = GlobalClassificationHead(
                global_dim=encoding_dim * num_views,
                num_classes=num_classes,
                hidden_dim=global_hidden_dim,
                dropout=dropout * 0.6
            )

    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        对所有视图进行编码

        Args:
            views: 列表，包含V个视图的输入特征
                   每个元素为 (batch_size, input_dim_v)
        Returns:
            encodings: 列表，包含V个视图的编码特征
                      每个元素为 (batch_size, encoding_dim)
        """
        assert len(views) == self.num_views, \
            f"Expected {self.num_views} views, got {len(views)}"

        encodings = []
        for v in range(self.num_views):
            encoding = self.encoders[v](views[v])
            encodings.append(encoding)

        return encodings

    def classify(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        对所有视图的编码进行分类

        Args:
            encodings: 列表，包含V个视图的编码特征
                      每个元素为 (batch_size, encoding_dim)
        Returns:
            logits: 列表，包含V个视图的分类logits
                   每个元素为 (batch_size, num_classes)
        """
        assert len(encodings) == self.num_views, \
            f"Expected {self.num_views} encodings, got {len(encodings)}"

        logits = []
        for v in range(self.num_views):
            logit = self.classification_heads[v](encodings[v])
            logits.append(logit)

        return logits

    def get_global_representation(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        获取全局表示 Z

        Args:
            encodings: 列表，包含V个视图的编码特征
                      每个元素为 (batch_size, encoding_dim)
        Returns:
            global_rep: 全局表示 Z (batch_size, V * encoding_dim)
        """
        return self.global_representation(encodings)

    def classify_global(self, global_rep: torch.Tensor) -> torch.Tensor:
        """
        使用全局分类头进行分类

        Args:
            global_rep: 全局表示 Z (batch_size, V * encoding_dim)
        Returns:
            logits: 全局分类logits (batch_size, num_classes)
        """
        if not self.use_global_head:
            raise ValueError("Global head is not enabled. Set use_global_head=True")
        return self.global_classifier(global_rep)

    def forward(self, views: List[torch.Tensor]) -> Dict:
        """
        前向传播：编码 + 视图分类 + 全局分类

        Args:
            views: 列表，包含V个视图的输入特征
                   每个元素为 (batch_size, input_dim_v)
        Returns:
            字典，包含:
                'encodings': V个视图的编码特征
                'view_logits': V个视图的分类logits
                'global_representation': 全局表示 Z
                'global_logits': 全局分类logits（如果启用）
        """
        # 编码所有视图
        encodings = self.encode(views)

        # 每个视图独立分类
        view_logits = self.classify(encodings)

        # 获取全局表示
        global_rep = self.get_global_representation(encodings)

        # 构建输出
        outputs = {
            'encodings': encodings,
            'view_logits': view_logits,
            'global_representation': global_rep
        }

        # 全局分类（如果启用）
        if self.use_global_head:
            global_logits = self.classify_global(global_rep)
            outputs['global_logits'] = global_logits

        return outputs

    def get_view_encoder(self, view_idx: int) -> MLPEncoder:
        """获取特定视图的编码器"""
        return self.encoders[view_idx]

    def get_view_classifier(self, view_idx: int) -> ClassificationHead:
        """获取特定视图的分类头"""
        return self.classification_heads[view_idx]
