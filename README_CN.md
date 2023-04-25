# SFighterAI

简体中文 | [English](README.md) | [Español](README_ES.md)

本项目基于深度强化学习训练了一个用于通关《街头霸王·二：冠军特别版》（Street Fighter II Special Champion Edition）关底 BOSS 的智能 AI 代理。该智能代理完全基于游戏画面（RGB 像素值）进行决策，在该项目给定存档中最后一关的第一轮对局可以取得 100% 胜率（实际上出现了“过拟合”现象，详见[运行测试](#running-tests)部分的讨论）。

### 文件结构

```bash
├───data
├───main
│   ├───logs
│   ├───trained_models
│   └───scripts
├───utils
│   └───scripts
```

游戏配置文件存储在 `data/` 文件夹下；项目的主要代码文件夹为 `main/`。其中，`logs/` 中包含了记录训练过程的终端文本和数据曲线（使用 Tensorboard 查看）；`trained_models/` 中包含了不同阶段的模型权重文件，可以用于在 `test.py` 中运行测试，观看智能代理在不同训练阶段学习到的对战策略的效果。

## 运行指南

本项目基于 Python 编程语言，主要使用了 [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等标准代码库。程序运行使用的 Python 版本为 3.8.10，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。以下配置过程已在 Windows 11 系统上测试通过。以下为控制台/终端（Console/Terminal/Shell）指令。

### 环境配置

```bash
# 创建 conda 环境，将其命名为 StreetFighterAI，Python 版本 3.8.10
conda create -n StreetFighterAI python=3.8.10
conda activate StreetFighterAI

# 安装 Python 代码库
cd [项目上级文件夹]/street-fighter-ai/main
pip install -r requirements.txt

# 运行程序脚本定位 gym-retro 游戏文件夹位置
cd ..
python .\utils\print_game_lib_folder.py
```

控制台输出文件夹路径后，将其复制到文件资源管理器中，跳转到对应路径。该文件夹为 gym-retro 下《街头霸王·二：冠军特别版》的游戏数据文件夹，其中包含了游戏 ROM 文件和数据配置文件。将本项目中 `data/` 文件夹下的 `Champion.Level12.RyuVsBison.state`、`data.json`、`metadata.json`、`scenario.json` 四个文件复制到该文件夹中，覆盖原有文件（可能需要提供管理员权限）。其中 `.state` 文件为《街头霸王·二：冠军特别版》难度四最后一关开局的游戏存档，三个 `.json` 文件为 gym-retro 配置文件，存储了游戏信息的内存地址（本项目只用到了其中的 [agent_hp] 与 [enemey_hp]，用于实时读取游戏人物的生命值）。

运行程序还需要《街头霸王·二：冠军特别版》（Street Fighter II Special Champion Edition）的游戏 ROM 文件（可以理解为游戏程序本身）。gym-retro 本身不提供游戏的 ROM 文件，需要自行通过合法途径获得。可以参考该[链接](https://wowroms.com/en/roms/sega-genesis-megadrive/street-fighter-ii-special-champion-edition-europe/26496.html)。

通过合法途径自行获得游戏 ROM 文件后，将其复制到前述 gym-retro 的游戏数据文件夹下，并重命名为`rom.md`。至此，环境配置准备工作完成。

注 1：如果想在游戏中手动抓取存档、寻找内存变量，可以使用 gym-retro integration ui，将`data/Gym Retro Integration.exe` 复制到前述 gym-retro 游戏数据文件夹下的上级菜单（上两级，`retro/` 文件夹下）即可。

注 2：如果想要录制智能代理的对战视频，还需要安装 [ffmpeg](https://ffmpeg.org/)。
```bash
conda install ffmpeg
```

### <a name="running-tests"></a>运行测试

环境配置完成后，可以在 `main/` 文件夹下运行 `test.py` 进行测试，实际体验智能代理在不同训练阶段的表现。

```bash
cd [项目上级文件夹]/street-fighter-ai/main
python test.py
```

模型权重文件存储在 `main/trained_models/` 文件夹下。其中 `ppo_ryu_2500000_steps_updated.zip` 是 `test.py` 默认使用的模型文件，该模型泛化性较好，有能力打通《街头霸王·二：冠军特别版》的最后一关。如果想要观看其他模型的表现，可以将 `test.py` 中的 `model_path` 变量修改为其他模型文件的路径。关于各训练阶段模型实际表现的观察描述如下：

* ppo_ryu_2000000_steps_updated: 刚开始出现过拟合现象，具有泛化能力但实力不太强。
* ppo_ryu_2500000_steps_updated: 接近最终过拟合状态，无法在最后一关第一轮中完全占据主导地位，但具有一定泛化能力。在最后一关三轮中有较高的获胜机会。
* ppo_ryu_3000000_steps_updated: 接近最终过拟合状态，几乎可以在最后一关第一轮中占据主导地位，胜率接近 100%，但泛化能力较弱。
* ppo_ryu_7000000_steps_updated: 过拟合，在最后一关第一轮中完全占据主导地位，胜率 100%，但泛化能力差。

### 训练模型

如果想要训练自己的模型，可以在 `main/` 文件夹下运行 `train.py`。

```bash
cd [项目上级文件夹]/street-fighter-ai/main
python train.py
```

### 查看曲线

项目中包含了训练过程的 Tensorboard 曲线图，可以使用 Tensorboard 查看其中的详细数据。推荐使用 VSCode 集成的 Tensorboard 插件直接查看（我爱你 VSCode！）。以下列出传统查看方法：

```bash
cd [项目上级文件夹]/street-fighter-ai/main
tensorboard --logdir=logs/
```

在浏览器中打开 Tensorboard 服务默认地址 `http://localhost:6006/`，即可查看训练过程的交互式曲线图。

## 鸣谢
本项目使用了 [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等开源代码库。感谢各位程序工作者对开源社区的贡献！

特别列出以下两篇对本项目启发作用很大的论文：

[1] [DIAMBRA Arena A New Reinforcement Learning Platform for Research and Experimentation](https://arxiv.org/abs/2210.10595)
这篇论文中关于格斗游戏深度强化学习模型超参数设置的经验总结非常有价值，对本项目的训练过程有很大的帮助。

[2] [Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf)
这篇论文中提出的“惩罚衰减”机制有效地解决了本次训练中智能代理在游戏中的“怯懈”（始终回避对手，不敢尝试攻击）问题，帮助非常大。
