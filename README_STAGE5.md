# 阶段 5: 量化部署

## 学习目标

1. 理解模型量化的原理
2. 掌握 4bit/8bit 量化技术
3. 实现模型合并和导出
4. 学习推理优化技巧

## 关键概念

### 模型量化

#### 什么是量化？

**量化**：将模型权重从高精度（FP32/FP16）转换为低精度（INT8/INT4），减少模型大小和显存占用。

#### 量化类型对比

| 精度 | 大小 | 显存 | 速度 | 精度损失 |
|------|------|------|------|---------|
| FP32 | 100% | 100% | 1x | 0% |
| FP16 | 50% | 50% | 2x | <1% |
| INT8 | 25% | 25% | 3-4x | 1-2% |
| INT4 | 12.5% | 12.5% | 4-6x | 2-5% |

#### 量化方法

**Post-Training Quantization (PTQ)**：
- 训练后量化
- 无需重新训练
- 快速但可能有精度损失

**Quantization-Aware Training (QAT)**：
- 训练时模拟量化
- 精度损失更小
- 需要重新训练

### LoRA 权重合并

#### 为什么要合并？

**LoRA 模式**：
- 优势：LoRA 权重很小（几 MB）
- 劣势：推理时需要加载基础模型 + LoRA 权重

**合并模式**：
- 优势：推理更快，部署更简单
- 劣势：模型文件变大

#### 合并过程

```python
# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(base_path)

# 2. 加载 LoRA 权重
lora_model = PeftModel.from_pretrained(base_model, lora_path)

# 3. 合并
merged_model = lora_model.merge_and_unload()

# 4. 保存
merged_model.save_pretrained(output_path)
```

### 推理优化技巧

#### 1. 使用量化

```python
# 4bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config
)
```

#### 2. 使用 Flash Attention

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2"
)
```

#### 3. 批量推理

```python
# 单条推理
outputs = model.generate(inputs, max_new_tokens=100)

# 批量推理（更快）
batch_inputs = tokenizer(prompts, padding=True, return_tensors="pt")
outputs = model.generate(**batch_inputs, max_new_tokens=100)
```

#### 4. KV Cache 优化

```python
# 启用 KV cache（默认开启）
outputs = model.generate(
    inputs,
    use_cache=True,  # 缓存 key/value，加速生成
    max_new_tokens=100
)
```

## 实践任务

### 任务 1: 运行部署脚本

```bash
python stage5_deploy.py
```

观察：
- 不同量化方式的显存占用
- 推理速度对比
- 模型大小对比

### 任务 2: 合并 LoRA 权重

如果你已经完成了阶段 4 的训练：

```bash
python stage5_deploy.py
```

脚本会自动：
1. 加载基础模型和 LoRA 权重
2. 合并权重
3. 保存合并后的模型到 `/app/outputs/merged`

### 任务 3: 测试量化效果

修改 `stage5_deploy.py`，测试不同量化配置：

```python
# 测试 4bit 量化
model = load_base_model_with_quantization(model_path, "4bit")

# 测试 8bit 量化
model = load_base_model_with_quantization(model_path, "8bit")

# 测试无量化
model = load_base_model_with_quantization(model_path, "none")
```

### 任务 4: 推理性能测试

运行 benchmark，对比不同配置的推理速度：

```python
benchmark_inference(model, tokenizer, prompt, num_runs=10)
```

## 部署方案选择

### 方案 1: LoRA 模式（推荐用于开发）

**优势**：
- LoRA 权重很小，易于版本管理
- 可以快速切换不同的 LoRA 权重
- 节省存储空间

**劣势**：
- 推理时需要加载两个模型
- 略慢于合并模式

**适用场景**：
- 开发和实验阶段
- 需要管理多个 LoRA 版本
- 存储空间有限

### 方案 2: 合并模式（推荐用于生产）

**优势**：
- 推理更快
- 部署更简单
- 只需一个模型文件

**劣势**：
- 模型文件较大
- 更新需要重新合并

**适用场景**：
- 生产环境部署
- 推理性能要求高
- 模型版本稳定

### 方案 3: 量化部署（推荐用于资源受限）

**优势**：
- 显存占用最小
- 推理速度快
- 适合边缘设备

**劣势**：
- 有一定精度损失
- 需要支持量化的硬件

**适用场景**：
- 显存有限（<8GB）
- 边缘设备部署
- 对精度要求不高

## 常见问题

### Q: 4bit 和 8bit 量化哪个更好？

**答**：
- **8bit**：精度损失小（1-2%），推荐优先使用
- **4bit**：显存占用更小，适合极端资源受限场景

### Q: 量化会影响微调效果吗？

**答**：
- 训练时不建议量化（精度损失）
- 推理时可以量化（影响很小）

### Q: 如何选择部署方案？

**答**：
- 开发阶段：LoRA 模式
- 生产环境：合并模式 + 8bit 量化
- 边缘设备：合并模式 + 4bit 量化

### Q: 合并后模型变大了怎么办？

**答**：
- 使用量化（8bit/4bit）
- 使用模型压缩技术
- 使用更小的基础模型

## 性能优化清单

- [ ] 使用量化（8bit/4bit）
- [ ] 启用 Flash Attention
- [ ] 使用批量推理
- [ ] 启用 KV Cache
- [ ] 合并 LoRA 权重（生产环境）
- [ ] 使用 GPU 推理
- [ ] 优化 max_new_tokens

## 里程碑检查点

- [ ] 理解量化的原理和类型
- [ ] 成功合并 LoRA 权重
- [ ] 测试不同量化方式
- [ ] 掌握推理优化技巧

## 下一步

完成本阶段后，进入 **阶段 6: 模型评估**，学习如何评估微调后的模型效果。
