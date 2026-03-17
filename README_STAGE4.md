# 阶段 4: 训练优化

## 学习目标

1. 理解训练超参数的作用
2. 掌握训练监控和日志
3. 实现完整的训练流程
4. 学习训练优化技巧

## 关键概念

### 训练超参数

#### Learning Rate (学习率)

- **含义**：控制参数更新的步长
- **范围**：通常 1e-5 到 5e-4
- **影响**：
  - 太大：训练不稳定，loss 震荡
  - 太小：训练太慢，可能陷入局部最优
- **推荐**：LoRA 微调使用 2e-4

#### Batch Size (批次大小)

- **含义**：每次训练使用的样本数量
- **影响**：
  - 越大：训练越稳定，但显存占用高
  - 越小：显存占用低，但训练不稳定
- **推荐**：根据显存调整，通常 4-16

#### Gradient Accumulation (梯度累积)

- **含义**：累积多个 batch 的梯度再更新
- **作用**：模拟大 batch size，节省显存
- **公式**：`有效 batch size = batch_size × gradient_accumulation_steps`
- **示例**：batch_size=4, accumulation=4 → 有效 batch_size=16

#### Epochs (训练轮数)

- **含义**：遍历整个数据集的次数
- **推荐**：
  - 小数据集（<1000）：3-5 epochs
  - 中等数据集（1000-10000）：2-3 epochs
  - 大数据集（>10000）：1-2 epochs

#### Warmup Steps (预热步数)

- **含义**：训练初期逐渐增大学习率
- **作用**：避免训练初期梯度过大导致不稳定
- **推荐**：总步数的 5-10%

### 训练优化技巧

#### 1. 混合精度训练 (FP16)

```yaml
training:
  fp16: true  # 使用 FP16，节省显存，加速训练
```

**优势**：
- 显存占用减半
- 训练速度提升 2-3 倍
- 精度损失可忽略

#### 2. 梯度检查点 (Gradient Checkpointing)

```yaml
training:
  gradient_checkpointing: true
```

**优势**：
- 大幅减少显存占用（50-70%）
- 代价：训练速度降低 20-30%

#### 3. 学习率调度

常见策略：
- **Linear**：线性衰减
- **Cosine**：余弦衰减
- **Constant with warmup**：预热后保持不变

#### 4. 早停 (Early Stopping)

监控验证集 loss，连续 N 步不下降则停止训练。

### 训练监控指标

#### Loss (损失)

- **Training Loss**：训练集损失，应持续下降
- **Validation Loss**：验证集损失，用于判断过拟合

**正常情况**：
```
Epoch 1: train_loss=2.5, val_loss=2.6
Epoch 2: train_loss=2.0, val_loss=2.1
Epoch 3: train_loss=1.5, val_loss=1.6
```

**过拟合**：
```
Epoch 1: train_loss=2.5, val_loss=2.6
Epoch 2: train_loss=2.0, val_loss=2.1
Epoch 3: train_loss=1.5, val_loss=2.3  ← val_loss 上升
```

#### Perplexity (困惑度)

- **公式**：`perplexity = exp(loss)`
- **含义**：模型对下一个词的不确定性
- **越低越好**

## 实践任务

### 任务 1: 运行训练脚本

```bash
python stage4_training.py
```

观察：
- 训练日志输出
- Loss 变化趋势
- 训练速度（steps/s）
- 显存占用

### 任务 2: 调整学习率

修改 `config/base_config.yaml`：

```yaml
training:
  learning_rate: 1.0e-4  # 尝试不同值：5e-5, 2e-4, 5e-4
```

观察学习率对训练的影响。

### 任务 3: 调整 Batch Size

```yaml
training:
  per_device_train_batch_size: 2  # 尝试 2, 4, 8
  gradient_accumulation_steps: 8  # 调整以保持有效 batch size
```

### 任务 4: 监控训练过程

训练时观察：
- Loss 是否持续下降
- 是否出现过拟合
- 训练速度是否合理

## 常见问题

### Q: 显存不足怎么办？

**解决方案**：
1. 减小 batch_size
2. 启用 gradient_checkpointing
3. 减小 max_length
4. 使用 8bit/4bit 量化

### Q: 训练太慢怎么办？

**解决方案**：
1. 增大 batch_size
2. 关闭 gradient_checkpointing
3. 使用更快的优化器（如 AdamW）
4. 减少 logging_steps

### Q: Loss 不下降怎么办？

**可能原因**：
1. 学习率太小 → 增大学习率
2. 数据质量差 → 检查数据
3. 模型配置错误 → 检查 LoRA 配置

### Q: 过拟合怎么办？

**解决方案**：
1. 增加训练数据
2. 减少训练轮数
3. 增大 lora_dropout
4. 使用数据增强

## 训练配置示例

### 小显存配置（8GB）

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
```

### 标准配置（16GB）

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  fp16: true
```

### 高性能配置（24GB+）

```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  gradient_checkpointing: false
  fp16: true
```

## 里程碑检查点

- [ ] 理解关键训练超参数
- [ ] 成功运行完整训练流程
- [ ] 掌握训练监控方法
- [ ] 学会调整参数优化训练

## 下一步

完成本阶段后，进入 **阶段 5: 量化部署**，学习如何优化和部署微调后的模型。
