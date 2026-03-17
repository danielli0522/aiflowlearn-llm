# 阶段 2: LoRA 原理

## 学习目标

1. 理解 LoRA 的核心思想
2. 掌握 LoRA 的关键参数
3. 实现 LoRA 模型配置
4. 对比全量微调 vs LoRA 微调

## 关键概念

### 什么是 LoRA？

**LoRA (Low-Rank Adaptation)** 是一种参数高效的微调技术。

**核心思想**：
- 冻结预训练模型的原始权重
- 在关键层（如注意力层）旁边添加低秩矩阵
- 只训练这些低秩矩阵，大幅减少可训练参数

### LoRA 的数学原理

对于权重矩阵 W，LoRA 添加一个低秩分解：

```
W' = W + ΔW
ΔW = B × A
```

其中：
- W: 原始权重（冻结）
- A: r × d 矩阵
- B: d × r 矩阵
- r: 秩（rank），远小于 d

### 关键参数

#### r (rank)
- **含义**：低秩矩阵的秩
- **范围**：通常 4-64
- **影响**：
  - 越大：可训练参数越多，表达能力越强
  - 越小：参数越少，训练越快，但可能欠拟合
- **推荐**：8-16 适合大多数任务

#### lora_alpha
- **含义**：缩放因子
- **作用**：控制 LoRA 权重的影响程度
- **公式**：`scaling = lora_alpha / r`
- **推荐**：通常设为 r 的 2 倍

#### lora_dropout
- **含义**：Dropout 概率
- **作用**：防止过拟合
- **推荐**：0.05-0.1

#### target_modules
- **含义**：应用 LoRA 的目标层
- **常见选择**：
  - `q_proj, k_proj, v_proj, o_proj`（注意力层）
  - `gate_proj, up_proj, down_proj`（FFN 层）
- **推荐**：先只用注意力层，效果不够再加 FFN

### 全量微调 vs LoRA

| 对比项 | 全量微调 | LoRA |
|--------|---------|------|
| 可训练参数 | 100% | <1% |
| 显存占用 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 存储成本 | 高（需保存完整模型） | 低（只保存 LoRA 权重） |
| 适用场景 | 大规模数据、充足资源 | 小数据、有限资源 |

## 实践任务

### 任务 1: 运行 LoRA 配置脚本

```bash
python stage2_lora.py
```

观察：
- 基础模型的参数量
- 应用 LoRA 后的可训练参数量
- 不同 r 值的参数对比

### 任务 2: 实验不同的 r 值

修改 `stage2_lora.py` 中的 r 值，观察参数量变化：

```python
lora_model, _ = apply_lora(base_model, r=4)   # 小配置
lora_model, _ = apply_lora(base_model, r=8)   # 标准配置
lora_model, _ = apply_lora(base_model, r=16)  # 大配置
```

### 任务 3: 理解 target_modules

尝试修改 target_modules，观察参数量变化：

```python
# 只用 Q/K/V
target_modules=["q_proj", "k_proj", "v_proj"]

# 加上输出层
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# 加上 FFN 层
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

## 里程碑检查点

- [ ] 理解 LoRA 的核心思想（低秩分解）
- [ ] 掌握 r、lora_alpha、lora_dropout 的含义
- [ ] 成功配置 LoRA 模型
- [ ] 理解全量微调 vs LoRA 的区别

## 常见问题

**Q: r 值应该设多大？**
A: 从 8 开始，效果不够再增大到 16 或 32。

**Q: lora_alpha 和 r 的关系？**
A: 通常 lora_alpha = 2 × r，保持 scaling = 2。

**Q: 为什么只训练注意力层？**
A: 注意力层是 Transformer 的核心，微调它们通常就能获得好效果。

## 下一步

完成本阶段后，进入 **阶段 3: 数据工程**，学习如何准备高质量的训练数据。
