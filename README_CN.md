# SFighterAI 街头霸王游戏智能代理

该项目基于深度强化学习训练了一个用于通关《街头霸王·二：冠军特别版》（Street Fighter II Special Champion Edition）关地 BOSS 的智能 AI 代理。该智能代理完全基于游戏画面（RGB 像素值）进行决策，在该项目给定存档中最后一关的第一轮对局可以取得 100% 胜率（实际上出现了“过拟合”现象，详见[结果]部分的讨论）。

### 文件结构

```bash
├───android
│   ├───app
│   │   └───src
│   └───gradle
├───doc_images
├───main
│   └───pose_data
│       └───train
│           ├───forwardhead
│           └───standard
```
