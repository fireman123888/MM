# 如何查看 MMatch 训练日志

## 🔍 实时方法

训练正在后台运行，日志保存在 `training_log.txt`

### 1. 查看完整日志
```bash
cat training_log.txt
```

### 2. 查看最新的日志（实时更新）
```bash
# Windows PowerShell
Get-Content training_log.txt -Wait -Tail 50

# Git Bash / WSL
tail -f training_log.txt
```

### 3. 查看特定内容

**查看每个epoch的结果**:
```bash
findstr "Epoch.*200" training_log.txt
# 或
grep "Epoch.*200" training_log.txt
```

**查看最佳模型保存信息**:
```bash
findstr "BEST" training_log.txt
# 或
grep "BEST" training_log.txt
```

**查看全局准确率**:
```bash
findstr "Global Acc" training_log.txt
# 或
grep "Global Acc" training_log.txt
```

**查看分布对齐信息**:
```bash
findstr "q_tilde" training_log.txt
# 或
grep "q_tilde" training_log.txt
```

### 4. 查看训练进度

**查看前100行（初始化信息）**:
```bash
head -100 training_log.txt
```

**查看最后50行（最新进度）**:
```bash
tail -50 training_log.txt
```

### 5. 搜索特定 Epoch

```bash
# 查看 Epoch 10 的结果
findstr "Epoch 10/200" training_log.txt

# 查看所有 epoch 摘要
findstr "Epoch [0-9]*/200:" training_log.txt
```

## 📊 Python 脚本查看

创建一个 Python 脚本来解析日志：

```python
# view_training_log.py
import re

def parse_log(log_file='training_log.txt'):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    epochs = []
    for i, line in enumerate(lines):
        # 查找 Epoch 摘要
        if 'Epoch' in line and '/200:' in line:
            print(line.strip())
            # 打印接下来的5行（包含详细信息）
            for j in range(1, 6):
                if i+j < len(lines):
                    print(lines[i+j].strip())
            print()

if __name__ == '__main__':
    parse_log()
```

运行：
```bash
python view_training_log.py
```

## 🎯 关键指标

训练日志中的关键信息：

```
Epoch X/200:
  训练 - Loss: X.XXXX (L_x: X.XXXX, L_u: X.XXXX), Global Acc: 0.XXXX, Mask Ratio: 0.XXX
  测试 - Loss: X.XXXX, Global Acc: 0.XXXX
  视图准确率 (测试):
    fou: 0.XXXX
    fac: 0.XXXX
    ...
  [BEST] Save best model (Global Acc: 0.XXXX)  <- 最佳模型
  分布对齐 q_tilde: min=0.XXXX, max=0.XXXX, std=0.XXXX
```

## 💡 快速检查命令

```bash
# Windows
powershell -Command "Get-Content training_log.txt | Select-Object -Last 100"

# 查看是否完成
findstr "训练完成" training_log.txt

# 查看最佳准确率
findstr "BEST" training_log.txt | findstr "Global Acc"
```

## 📈 训练完成后

训练完成后查看最终结果：
```bash
# 查看最后100行
tail -100 training_log.txt

# 查看训练完成信息
grep "训练完成" training_log.txt -A 10
```

## 🔄 当前状态

文件位置: `C:\Users\23126\Downloads\MMatch-try\training_log.txt`

训练正在后台运行，你可以随时用上面的命令查看进度！
