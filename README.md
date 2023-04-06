# SFighterAI 街头霸王游戏智能代理

本项目基于深度强化学习训练了一个用于通关《街头霸王·二：冠军特别版》（Street Fighter II Special Champion Edition）关地 BOSS 的智能 AI 代理。该智能代理完全基于游戏画面（RGB 像素值）进行决策，在该项目给定存档中最后一关的第一轮对局可以取得 100% 胜率（实际上出现了“过拟合”现象，详见[结果]部分的讨论）。

### 文件结构

```bash
├───main
│   ├───logs
│   ├───trained_models
│   └───scripts
├───utils
│   └───scripts
```

项目的主要文件夹为 `main/`。其中，`logs/` 中包含了记录训练过程的终端文本和数据曲线（使用 Tensorboard 查看）；`trained_models/` 中包含了不同阶段的模型权重文件，可以用于在 `test.py` 中运行测试，观看智能代理在不同训练阶段学习到的对战策略的效果。

## 运行指南

本项目基于 Python 编程语言，主要使用了 [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等标准代码库。程序运行使用的 Python 版本为 3.8.10，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。

```bash
# 创建 conda 环境，将其命名为 StreetFighterAI，Python 版本 3.8.10
conda create -n StreetFighterAI python=3.8.10
conda activate StreetFighterAI

# 注：conda 在苹果 M1 芯片（Apple Silicon）上对 python 向下支持到 3.8.11，使用以下指令创建环境：
# conda create -n StreetFighterAI python=3.8.11

# 安装 Python 代码库
cd [project_dir]/street-fighter-ai/main
pip install -r requirements.txt
```
