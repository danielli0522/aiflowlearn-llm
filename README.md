# AIFlowLearn LLM 微调实战项目

> 基于 Qwen2.5-0.5B 的 LLM 微调教学项目
> 使用 digital-twin 数据训练个人数字分身

## 项目结构

本项目包含 6 个学习阶段，每个阶段对应一个 git 分支：

```
main (基础环境)
├── stage-1-basics    # 微调基础：对比通用模型 vs 微调模型
├── stage-2-lora      # LoRA 原理：参数配置对比实验
├── stage-3-data      # 数据工程：数据处理和格式转换
├── stage-4-training  # 训练优化：完整训练流程
├── stage-5-deploy    # 量化部署：模型量化和推理
└── stage-6-eval      # 模型评估：效果评估和对比
```

## 环境要求

- **Python**: 3.10+
- **GPU**: 6GB+ 显存（推荐 8GB+）
- **磁盘**: 5GB+ 可用空间
- **系统**: Linux / macOS

## 快速开始

### 使用 Docker（推荐）

```bash
docker-compose up -d
docker exec -it aiflowlearn-llm bash
```

首次启动会自动下载 Qwen2.5-0.5B 模型（约 1GB），需要 2-3 分钟。

## 学习路径

### Stage 1: 微调基础
```bash
git checkout stage-1-basics
cat README_STAGE1.md  # 阅读学习指南
python stage1_basics.py
```

**学习目标**：理解预训练 vs 微调、模型加载、基础推理

### Stage 2: LoRA 原理
```bash
git checkout stage-2-lora
cat README_STAGE2.md
python stage2_lora.py
```

**学习目标**：理解 LoRA 原理、参数配置、对比全量微调

### Stage 3: 数据工程
```bash
git checkout stage-3-data
cat README_STAGE3.md
python stage3_data.py
```

**学习目标**：数据清洗、Alpaca 格式、质量检查

### Stage 4: 训练优化
```bash
git checkout stage-4-training
cat README_STAGE4.md
python stage4_training.py
```

**学习目标**：训练超参数、监控指标、优化技巧

### Stage 5: 量化部署
```bash
git checkout stage-5-deploy
cat README_STAGE5.md
python stage5_deploy.py
```

**学习目标**：模型量化、权重合并、推理优化

### Stage 6: 模型评估
```bash
git checkout stage-6-eval
cat README_STAGE6.md
python stage6_eval.py
```

**学习目标**：评估指标、自动评估、人工评估、效果对比

## 模型说明

- **基础模型**: Qwen2.5-0.5B-Instruct
- **模型大小**: ~1GB
- **微调方法**: LoRA
- **训练时间**: 约 30 分钟（1000 条数据）
