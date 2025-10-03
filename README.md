# VAE (Variational Autoencoder) 实现

一个功能完整的变分自编码器(VAE)实现，支持多种模型架构、损失函数和训练策略。

## 项目特性

### 🏗️ 模型架构
- **MLP VAE**: 基于全连接层的经典VAE实现
- **CNN VAE**: 基于卷积神经网络的VAE，适用于图像数据
- **Progressive VAE**: 渐进式训练的VAE，支持多分辨率训练
- **Residual Progressive VAE**: 残差渐进式VAE

### 📊 数据集支持
- **MNIST**: 手写数字数据集
- **FreyFaces**: 人脸数据集 (28x20 灰度图像)

### 🔧 损失函数
- **重建损失**:
  - Binary Cross Entropy (BCE)
  - Gaussian Negative Log-Likelihood
- **KL散度**:
  - 解析形式 (Analytic)
  - 蒙特卡洛采样 (Monte Carlo)

### 📈 实验管理
- 自动实验目录创建和配置保存
- 训练日志记录
- 模型检查点保存
- 可视化结果生成 (重建图像、生成样本)
- FID评估指标

## 安装依赖

```bash
pip install torch torchvision scipy numpy matplotlib
```

## 快速开始

### 1. 训练标准VAE

```bash
# MNIST数据集 + MLP模型 + BCE重建损失
python train_vae.py --dataset mnist --model mlp --recon bce --epochs 30

# FreyFaces数据集 + MLP模型 + Gaussian重建损失
python train_vae.py --dataset freyfaces --model mlp --recon gaussian --epochs 20

# MNIST数据集 + CNN模型
python train_vae.py --dataset mnist --model cnn --recon bce --epochs 30
```

### 2. 训练Progressive VAE

```bash
# Progressive VAE训练
python train_provae.py --epochs_per_stage 10 --start_res 4 --final_res 32
```

## 命令行参数

### train_vae.py 参数

#### 基本参数
- `--exp_name`: 实验名称 (默认: 'vae')
- `--data_path`: 数据路径 (默认: './data')
- `--dataset`: 数据集选择 ['mnist', 'freyfaces']
- `--model`: 模型类型 ['mlp', 'cnn']

#### 模型参数
- `--input_dim`: 输入维度 (仅MLP，默认自动设置)
- `--hidden_dim`: 隐藏层维度 (默认: 400)
- `--z_dim`: 潜在空间维度 (默认: 20)

#### 训练参数
- `--batch_size`: 批次大小 (默认: 128)
- `--lr`: 学习率 (默认: 1e-3)
- `--epochs`: 训练轮数 (默认: 30)
- `--seed`: 随机种子 (默认: 42)

#### 损失函数参数
- `--recon`: 重建损失类型 ['bce', 'gaussian']
- `--kl`: KL散度计算方式 ['analytic', 'mc']
- `--beta`: KL散度权重 (默认: 1.0)
- `--reduction`: 损失聚合方式 ['sum', 'mean']
- `--sigma`: Gaussian重建损失的标准差 (默认: 1.0)
- `--sigma_learnable`: 使sigma参数可学习
- `--include_const`: Gaussian NLL包含常数项

### train_provae.py 参数

- `--epochs_per_stage`: 每阶段训练轮数 (默认: 10)
- `--fadein_ratio`: 渐入比例 (默认: 0.5)
- `--start_res`: 起始分辨率 (默认: 4)
- `--final_res`: 最终分辨率 (默认: 32)
- `--base_ch`: 基础通道数 (默认: 128)
- `--min_ch`: 最小通道数 (默认: 16)

## 项目结构
VAE/
├── models/                    # 模型定义
│   ├── VAE.py                # 标准MLP VAE
│   ├── CNN_VAE.py            # 卷积VAE
│   ├── ProVAE.py             # 渐进式VAE
│   └── ResProVAE.py          # 残差渐进式VAE
├── train_vae.py              # 标准VAE训练脚本
├── train_provae.py           # 渐进式VAE训练脚本
├── loss.py                   # 损失函数实现
├── utils.py                  # 工具函数
├── experiments/              # 实验结果目录
└── data/                     # 数据目录