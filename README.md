# BERT 深度学习性能优化项目

这个项目旨在展示和实现深度学习模型（特别是BERT模型）的各种性能优化技术，包括混合精度训练、ONNX转换，以及使用TensorRT加速推理。本项目还包括将优化后的模型部署到AWS云端进行实时预测的完整流程。

## 📚 项目概述

BERT (Bidirectional Encoder Representations from Transformers) 是一种强大的预训练语言模型，但其复杂的结构和参数量使其在资源受限环境中的应用受到限制。本项目通过实现一系列优化技术，在保持模型准确性的同时，显著提高其训练和推理性能。

### 实现的优化技术

- **混合精度训练**：使用FP16加速计算，减少显存占用
- **ONNX转换**：将模型转换为开放神经网络交换格式以实现跨平台兼容
- **TensorRT优化**：使用NVIDIA的TensorRT库加速推理过程
- **AWS部署**：完整的模型云端部署流程，实现实时预测服务

### 主要特点

- 🚀 完整的训练到部署流程
- 📊 详细的性能对比分析
- 🔄 各种优化技术的组合应用
- ☁️ AWS SageMaker部署方案
- 📈 可视化性能报告

## 🛠️ 安装与准备

### 系统要求

- Python 3.9+
- CUDA 11.2+ (用于GPU加速)
- NVIDIA GPU (建议RTX系列，用于TensorRT优化)
- 8GB+ RAM
- Windows/Linux/MacOS (TensorRT仅支持Windows和Linux)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/YingzuoLiu/bert-optimization-project.git
cd bert-optimization-project

# 创建虚拟环境
conda create -n dl_optimization python=3.9
conda activate dl_optimization

# 安装依赖
pip install -r requirements.txt

# 安装TensorRT (根据您的CUDA版本选择适当的TensorRT版本)
# 请参考NVIDIA官方文档: https://developer.nvidia.com/tensorrt
```

## 🔄 使用方法

### 运行完整流程

执行主脚本以运行整个优化流程：

```bash
python main_pytorch.py
```

如果您想部署到AWS，请提供您的AWS凭证：

```bash
python main_pytorch.py --aws_account_id YOUR_AWS_ACCOUNT_ID --aws_role_arn YOUR_AWS_ROLE_ARN
```

### 运行单独步骤

您也可以单独运行各优化步骤：

```bash
# 只运行基准测试
python benchmark_model_pytorch.py

# 只运行混合精度训练
python optimization_mixed_precision_pytorch.py

# 只运行ONNX和TensorRT优化
python onnx_tensorrt_pytorch.py

# 只运行性能对比
python performance_comparison_pytorch.py
```

## ☁️ AWS部署与资源管理

### 部署到AWS

单独执行部署脚本：

```bash
python aws_deployment_pytorch.py --model_path ./saved_models/bert_mixed_precision --aws_region us-east-1 --aws_account_id YOUR_AWS_ACCOUNT_ID --role_arn YOUR_AWS_ROLE_ARN
```

或者使用分步部署方式：

```bash
# 第一步：准备部署包
python aws_deployment_prepare.py --aws_account_id YOUR_AWS_ACCOUNT_ID --aws_region us-east-1

# 第二步：完成部署
python aws_deployment_finish.py --aws_account_id YOUR_AWS_ACCOUNT_ID --aws_region us-east-1 --role_arn YOUR_AWS_ROLE_ARN
```

### 清理AWS资源（重要！）

为防止产生不必要的费用，请在使用完毕后清理所有AWS资源：

```bash
# 通过主脚本清理
python main_pytorch.py --cleanup_aws --aws_region us-east-1

# 或直接使用清理脚本
python aws_cleanup.py --aws_region us-east-1 --model_name bert-sentiment-analysis --repository_name bert-sentiment-analysis
```

这将删除：
- SageMaker端点、配置和模型
- ECR仓库及其镜像
- 相关的CloudWatch日志组

## 📊 结果分析

执行完整流程后，您可以查看以下结果：

- **性能报告**：`reports/performance_report.html`
- **性能图表**：`performance_plots/`目录下的各种对比图表
- **优化模型**：`saved_models/`目录中保存的各类优化模型

基于项目中的图表分析，我们可以看到：
- 混合精度训练比基准模型的训练速度提升约48%
- 使用TensorRT可以将推理速度提升约92%
- 模型准确率保持在92-93%左右，与基准模型相当
- 所有优化方法都保持了高准确率，同时显著提升了推理性能

## 📁 项目结构

```
bert-optimization-project/
│
├── main_pytorch.py                # 主执行脚本
├── benchmark_model_pytorch.py     # 基准模型构建与训练
├── optimization_mixed_precision_pytorch.py  # 混合精度训练
├── onnx_tensorrt_pytorch.py       # ONNX和TensorRT优化
├── performance_comparison_pytorch.py  # 性能对比分析
├── aws_deployment_pytorch.py      # AWS一体化部署脚本
├── aws_deployment_prepare.py      # AWS部署准备脚本
├── aws_deployment_finish.py       # AWS部署完成脚本
├── aws_cleanup.py                 # AWS资源清理脚本
│
├── data/                          # 数据目录
├── saved_models/                  # 保存的模型目录
├── performance_plots/             # 性能图表目录
├── reports/                       # 性能报告目录
└── deployment_package/            # 部署包目录
```

## 📝 许可证

此项目基于MIT许可证开源 - 详情请参阅 [LICENSE](LICENSE) 文件

## 👥 贡献

欢迎贡献代码或提出改进建议！请遵循以下步骤：

1. Fork 这个仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的改动 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个 Pull Request

---

感谢使用本项目！希望它能帮助您优化深度学习模型的性能。
