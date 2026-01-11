# MMatch 分布对齐 (Distribution Alignment) 实现说明

## 核心概念

分布对齐 (DA) 是 MMatch 中的关键技术，用于解决半监督学习中的**类别分布偏差**问题。

### 问题
在半监督学习中，模型可能会对某些类别产生偏好，导致预测分布与真实分布不一致。

### 解决方案
维护一个类别概率的**移动平均** $\tilde{q}$，并使用它来调整预测分布。

## 实现细节

### 1. 移动平均 $\tilde{q}$ 的维护

```python
class DistributionAlignment:
    def __init__(self, num_classes, momentum=0.999):
        self.num_classes = num_classes
        self.momentum = momentum
        # 初始化为均匀分布
        self.q_tilde = torch.ones(num_classes) / num_classes
```

**更新公式**：
$$\tilde{q}_t = \alpha \cdot \tilde{q}_{t-1} + (1-\alpha) \cdot \bar{q}_t$$

其中：
- $\tilde{q}_t$ 是时刻 $t$ 的移动平均
- $\bar{q}_t$ 是当前批次的平均预测分布
- $\alpha$ 是动量参数（默认 0.999）

```python
def update(self, probs):
    # 计算当前批次的平均分布
    q_batch = probs.mean(dim=0)  # (num_classes,)

    # 移动平均更新
    self.q_tilde = self.momentum * self.q_tilde + (1 - self.momentum) * q_batch.detach()
```

### 2. 分布对齐操作

**论文公式**：
$$q = \text{Normalize}\left(\frac{q}{\tilde{q}}\right)$$

其中 $\text{Normalize}(a)_i = \frac{a_i}{\sum_j a_j}$

**实现**：
```python
def align(self, probs):
    # 将 q_tilde 移到相同设备
    q_tilde = self.q_tilde.to(probs.device)

    # 避免除以零
    q_tilde = torch.clamp(q_tilde, min=1e-6)

    # 缩放: q / q_tilde
    scaled = probs / q_tilde.unsqueeze(0)

    # 归一化: Normalize(a)_i = a_i / Σ_j a_j
    aligned_probs = scaled / scaled.sum(dim=1, keepdim=True)

    return aligned_probs
```

### 3. 在训练中的使用

```python
# 1. 获取无标签数据的预测概率
unlabeled_probs = F.softmax(unlabeled_logits, dim=1)

# 2. 更新移动平均
da_module.update(unlabeled_probs)

# 3. 应用分布对齐
aligned_probs = da_module.align(unlabeled_probs)

# 4. 锐化（增强确定性）
pseudo_labels = sharpen(aligned_probs, T=0.5)

# 5. 使用伪标签进行训练
max_probs, pseudo_targets = torch.max(pseudo_labels, dim=1)
mask = max_probs >= threshold  # 只使用高置信度样本
```

## 完整的 MMatch 训练流程

### 监督损失 $\mathcal{L}_x$

$$\mathcal{L}_x = \sum_{v=1}^{V} \mathcal{L}_{x}^{v} + \mathcal{L}_{x}^{g}$$

其中：
- $\mathcal{L}_{x}^{v} = \frac{1}{B} \sum_{i=1}^{B} H(y_i, p_i^v)$ - 视图级监督损失
- $\mathcal{L}_{x}^{g} = \frac{1}{B} \sum_{i=1}^{B} H(y_i, q_i)$ - 全局级监督损失

### 无监督损失 $\mathcal{L}_u$

$$\mathcal{L}_u = \sum_{v=1}^{V} \mathcal{L}_{u}^{v} + \mathcal{L}_{u}^{g}$$

其中伪标签通过分布对齐生成：

1. 预测概率：$q = \text{softmax}(f(x_u))$
2. 分布对齐：$q' = \text{Normalize}(q / \tilde{q})$
3. 锐化：$\hat{q} = \text{Normalize}(q'^{1/T})$
4. 伪标签：$\hat{y} = \arg\max \hat{q}$
5. 置信度过滤：只使用 $\max(\hat{q}) \geq \tau$ 的样本

### 总损失

$$\mathcal{L} = \mathcal{L}_x + \lambda_u \cdot \mathcal{L}_u$$

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `momentum` | 0.999 | DA 移动平均动量 |
| `lambda_u` | 1.0 | 无监督损失权重 |
| `threshold` | 0.95 | 伪标签置信度阈值 |
| `temperature` | 0.5 | 锐化温度（越小越锐化） |

## 运行方法

```bash
python train_mmatch.py
```

## 预期效果

相比纯监督训练：
- ✅ 更好地利用无标签数据
- ✅ 缓解类别分布偏差
- ✅ 提高模型泛化能力
- ✅ 在小样本场景下效果更明显

## 输出示例

```
Epoch 10/200:
  训练 - Loss: 5.2341 (L_x: 3.1234, L_u: 2.1107), Global Acc: 0.8500, Mask Ratio: 0.723
  测试 - Loss: 4.5678, Global Acc: 0.8900
  视图准确率 (测试):
    fou: 0.7500
    fac: 0.8200
    ...
  分布对齐 q_tilde: min=0.0823, max=0.1156, std=0.0102
```

其中：
- `L_x`: 监督损失（有标签数据）
- `L_u`: 无监督损失（无标签数据的伪标签）
- `Mask Ratio`: 通过置信度阈值的无标签样本比例
- `q_tilde`: 当前的移动平均分布统计

## 关键改进点

1. **分布对齐**：自动调整类别偏差
2. **置信度过滤**：只使用高质量伪标签
3. **锐化操作**：增强预测确定性
4. **多视图协同**：所有视图共同生成伪标签

## 调试技巧

1. 监控 `Mask Ratio`：
   - 太低（<0.3）：降低 threshold
   - 太高（>0.9）：提高 threshold

2. 监控 `q_tilde` 的 std：
   - 太小（<0.01）：说明分布过于均匀，可能需要调整 momentum
   - 太大（>0.05）：说明分布偏差严重，DA 正在起作用

3. 对比 L_x 和 L_u：
   - L_u >> L_x：伪标签质量可能不好，提高 threshold
   - L_u << L_x：可以增加 lambda_u
