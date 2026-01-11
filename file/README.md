# 多视图编码器实现

基于 PyTorch 实现的多视图编码器系统，为 V 个视图分别创建独立的编码器和分类头，同时支持全局表示和全局分类。

## 特性

- 为每个视图创建独立的 MLP 编码器 (f^v)
- 为每个视图创建独立的分类头 (h^v)
- **全局表示 Z**: 将所有视图的特征拼接（concatenate）形成全局表示
- **全局分类头 H_Φ**: 对全局表示进行分类
- 支持不同视图使用不同的输入维度
- 统一的编码输出维度
- 支持 BatchNorm 和 Dropout
- 灵活的 MLP 隐藏层配置

## 文件结构

```
├── multi_view_encoder.py              # 主要实现
├── example_usage.py                   # 基本使用示例
├── example_global_representation.py   # 全局表示示例
└── README.md                          # 文档
```

## 快速开始

### 安装依赖

```bash
pip install torch
```

### 基本使用

```python
import torch
from multi_view_encoder import MultiViewEncoder

# 创建多视图编码器（带全局分类头）
model = MultiViewEncoder(
    num_views=3,                    # 3个视图
    input_dims=[512, 256, 128],     # 每个视图的输入维度
    hidden_dims=[256, 128],         # MLP隐藏层
    encoding_dim=64,                # 编码后的统一维度
    num_classes=10,                 # 分类任务类别数
    dropout=0.5,
    use_batch_norm=True,
    use_global_head=True,           # 启用全局分类头
    global_hidden_dim=128           # 全局分类头的隐藏层维度
)

# 准备输入数据（3个视图）
views = [
    torch.randn(32, 512),  # 视图1
    torch.randn(32, 256),  # 视图2
    torch.randn(32, 128),  # 视图3
]

# 前向传播
outputs = model(views)
encodings = outputs['encodings']              # 各视图编码特征
view_logits = outputs['view_logits']          # 各视图分类logits
global_rep = outputs['global_representation'] # 全局表示 Z
global_logits = outputs['global_logits']      # 全局分类logits
```

## 核心组件

### 1. MLPEncoder
单个视图的 MLP 编码器，将输入特征映射到统一的编码空间。

**参数:**
- `input_dim`: 输入特征维度
- `hidden_dims`: 隐藏层维度列表，如 `[512, 256]`
- `output_dim`: 输出编码维度
- `dropout`: Dropout 概率（默认 0.5）
- `use_batch_norm`: 是否使用 BatchNorm（默认 True）

### 2. ClassificationHead
单个视图的分类头，将编码特征映射到类别空间。

**参数:**
- `input_dim`: 输入编码维度
- `num_classes`: 类别数量
- `dropout`: Dropout 概率（默认 0.3）

### 3. GlobalRepresentation
全局表示模块，将所有视图的编码拼接形成全局表示 Z。

**功能:**
- 输入: V 个编码，每个维度为 `encoding_dim`
- 输出: 全局表示 Z，维度为 `V * encoding_dim`
- 操作: `torch.cat(encodings, dim=1)`

### 4. GlobalClassificationHead (H_Φ)
全局分类头，对全局表示进行分类。

**参数:**
- `global_dim`: 全局表示的维度（V * encoding_dim）
- `num_classes`: 类别数量
- `hidden_dim`: 可选的隐藏层维度
- `dropout`: Dropout 概率（默认 0.3）

### 5. MultiViewEncoder
多视图编码器系统，管理多个视图的编码器、分类头和全局表示。

**参数:**
- `num_views`: 视图数量 V
- `input_dims`: 每个视图的输入维度列表
- `hidden_dims`: MLP 隐藏层维度
- `encoding_dim`: 统一的编码维度
- `num_classes`: 分类任务类别数
- `dropout`: Dropout 概率
- `use_batch_norm`: 是否使用 BatchNorm
- `use_global_head`: 是否使用全局分类头
- `global_hidden_dim`: 全局分类头的隐藏层维度（可选）

**主要方法:**
- `encode(views)`: 编码所有视图
- `classify(encodings)`: 对各视图编码进行分类
- `get_global_representation(encodings)`: 获取全局表示 Z
- `classify_global(global_rep)`: 使用全局分类头分类
- `forward(views)`: 完整的前向传播
- `get_view_encoder(view_idx)`: 获取特定视图的编码器
- `get_view_classifier(view_idx)`: 获取特定视图的分类头

## 使用示例

运行示例代码：

```bash
# 基本使用示例
python example_usage.py

# 全局表示示例
python example_global_representation.py
```

### 示例 1: 基本前向传播（带全局表示）

```python
model = MultiViewEncoder(
    num_views=3,
    input_dims=[512, 256, 128],
    hidden_dims=[256, 128],
    encoding_dim=64,
    num_classes=10,
    use_global_head=True,
    global_hidden_dim=128
)

views = [torch.randn(32, dim) for dim in [512, 256, 128]]
outputs = model(views)

# 输出包含:
# - encodings: 各视图编码 (3个，每个 [32, 64])
# - view_logits: 各视图分类 (3个，每个 [32, 10])
# - global_representation: 全局表示 Z [32, 192]
# - global_logits: 全局分类 [32, 10]
```

### 示例 2: 训练循环（带全局损失）

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
outputs = model(views)
view_logits = outputs['view_logits']
global_logits = outputs['global_logits']

# 计算各视图损失
view_loss = sum(criterion(logits, labels) for logits in view_logits) / num_views

# 计算全局损失
global_loss = criterion(global_logits, labels)

# 总损失
total_loss = view_loss + global_loss
total_loss.backward()
optimizer.step()
```

### 示例 3: 全局表示推理

```python
model.eval()
with torch.no_grad():
    outputs = model(views)
    global_logits = outputs['global_logits']

    # 使用全局表示的预测
    global_probs = torch.softmax(global_logits, dim=1)
    predictions = torch.argmax(global_probs, dim=1)
```

## 架构说明

```
                    视图级别编码和分类
视图1 ─→ f^1 (MLP) ─→ 编码1(z^1) ─→ h^1 ─→ logits1
                           │
视图2 ─→ f^2 (MLP) ─→ 编码2(z^2) ─→ h^2 ─→ logits2
                           │
视图3 ─→ f^3 (MLP) ─→ 编码3(z^3) ─→ h^3 ─→ logits3
                           │
                           ↓
                   Concatenate (拼接)
                           ↓
                  全局表示 Z = [z^1, z^2, z^3]
                           ↓
                  全局分类头 H_Φ
                           ↓
                    global_logits
```

**关键点:**
- 每个视图有独立的编码器 f^v（MLP）和分类头 h^v
- 所有视图的编码维度相同（便于拼接）
- 全局表示 Z = concatenate([z^1, z^2, ..., z^V])
- Z 的维度 = V × encoding_dim
- 全局分类头 H_Φ 对 Z 进行分类

## 输出说明

`forward()` 方法返回一个字典，包含：

| 键 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `encodings` | List[Tensor] | V × [B, encoding_dim] | 各视图的编码特征 |
| `view_logits` | List[Tensor] | V × [B, num_classes] | 各视图的分类输出 |
| `global_representation` | Tensor | [B, V × encoding_dim] | 全局表示 Z |
| `global_logits` | Tensor | [B, num_classes] | 全局分类输出（如启用） |

其中 B = batch_size，V = num_views

## 扩展建议

1. **添加视图融合模块**: 可以在编码层添加注意力机制或其他融合策略
2. **添加对比学习损失**: 在编码空间中对齐不同视图的表示
3. **动态权重**: 根据每个视图的质量动态调整损失权重
4. **共享参数**: 可以选择在某些层共享参数以减少模型大小

## 依赖

- Python >= 3.7
- PyTorch >= 1.8.0

## License

MIT
