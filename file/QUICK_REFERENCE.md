# 多视图编码器快速参考

## 架构概览

```
输入: V个视图，每个视图有不同的维度
  ↓
[视图编码] 每个视图独立编码
  f^1: view1 → z^1 (encoding_dim)
  f^2: view2 → z^2 (encoding_dim)
  ...
  f^V: viewV → z^V (encoding_dim)
  ↓
[视图分类] 每个编码独立分类
  h^1: z^1 → logits1 (num_classes)
  h^2: z^2 → logits2 (num_classes)
  ...
  h^V: z^V → logitsV (num_classes)
  ↓
[全局表示] 拼接所有编码
  Z = [z^1, z^2, ..., z^V]
  维度: V * encoding_dim
  ↓
[全局分类] 全局分类头
  H_Φ: Z → global_logits (num_classes)
```

## 核心公式

### 视图编码
```
z^v = f^v(x^v)
```
其中:
- x^v: 第v个视图的输入
- f^v: 第v个视图的MLP编码器
- z^v: 第v个视图的编码特征

### 全局表示
```
Z = [z^1, z^2, ..., z^V]
```
使用 concatenate 操作拼接

### 分类
```
视图分类: y^v = h^v(z^v)
全局分类: y_global = H_Φ(Z)
```

## 代码示例

### 创建模型
```python
from multi_view_encoder import MultiViewEncoder

model = MultiViewEncoder(
    num_views=3,                 # V个视图
    input_dims=[512, 256, 128],  # 各视图输入维度
    hidden_dims=[256, 128],      # MLP隐藏层
    encoding_dim=64,             # 统一编码维度
    num_classes=10,              # 类别数
    use_global_head=True,        # 启用全局分类
    global_hidden_dim=128        # 全局分类隐藏层
)
```

### 前向传播
```python
# 输入
views = [
    torch.randn(32, 512),  # 视图1
    torch.randn(32, 256),  # 视图2
    torch.randn(32, 128),  # 视图3
]

# 前向传播
outputs = model(views)

# 输出
encodings = outputs['encodings']              # List: 3 × [32, 64]
view_logits = outputs['view_logits']          # List: 3 × [32, 10]
global_rep = outputs['global_representation'] # Tensor: [32, 192]
global_logits = outputs['global_logits']      # Tensor: [32, 10]
```

### 训练
```python
criterion = torch.nn.CrossEntropyLoss()

# 视图损失
view_loss = sum(
    criterion(logits, labels)
    for logits in outputs['view_logits']
) / num_views

# 全局损失
global_loss = criterion(outputs['global_logits'], labels)

# 总损失
total_loss = view_loss + global_loss
total_loss.backward()
```

## 维度变化

以 3 视图为例:

| 阶段 | 视图1 | 视图2 | 视图3 | 全局 |
|------|-------|-------|-------|------|
| 输入 | [B, 512] | [B, 256] | [B, 128] | - |
| 编码 | [B, 64] | [B, 64] | [B, 64] | - |
| 视图分类 | [B, 10] | [B, 10] | [B, 10] | - |
| 全局表示 | - | - | - | [B, 192] |
| 全局分类 | - | - | - | [B, 10] |

B = batch_size

## 损失函数

### 单视图损失
```python
L_v = CrossEntropy(h^v(z^v), y)
```

### 总损失 (示例)
```python
L_total = (1/V) * Σ L_v + L_global

其中:
L_v: 第v个视图的分类损失
L_global: 全局分类损失
```

## 关键特性

✅ **独立编码**: 每个视图有自己的编码器
✅ **独立分类**: 每个视图有自己的分类头
✅ **全局融合**: 通过拼接实现多视图融合
✅ **灵活配置**: 可选择是否使用全局分类头
✅ **可扩展**: 易于添加注意力机制、对比学习等

## 应用场景

1. **多模态学习**: 图像 + 文本 + 音频
2. **医学影像**: 不同角度/模态的医学图像
3. **推荐系统**: 用户行为 + 商品属性 + 社交网络
4. **视频分析**: RGB + 光流 + 音频
5. **生物信息**: 基因 + 蛋白质 + 表型数据
