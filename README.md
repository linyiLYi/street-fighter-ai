# SFighterAI

[简体中文](README_CN.md) | English | [Español](README_ES.md)

This project is an AI agent trained using deep reinforcement learning to beat the final boss in the game "Street Fighter II: Special Champion Edition". The AI agent makes decisions based solely on the game screen's RGB pixel values. In the provided save state, the agent achieves a 100% win rate in the first round of the final level (overfitting occurs, see the [Running Tests](#running-tests) section for discussion).

### File Structure

```bash
├───data
├───main
│   ├───logs
│   ├───trained_models
│   └───scripts
├───utils
│   └───scripts
```

The game configuration files are stored in the `data/` folder, and the main project code is in the `main/` folder. Within `main/`, the `logs/` folder contains terminal/console outputs and data curves recording the training process (viewable with Tensorboard), while the `trained_models/` folder contains model weights from different stages. These weights can be used for running tests in `test.py` to observe the performance of the AI agent's learned strategies at different training stages.

## Running Guide

This project is based on the Python programming language and primarily utilizes standard libraries like [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The Python version used is 3.8.10, and it is recommended to use [Anaconda](https://www.anaconda.com) to configure the Python environment. The following setup process has been tested on Windows 11. Below are console/terminal/shell commands.

### Environment Setup

```bash
# Create a conda environment named StreetFighterAI with Python version 3.8.10
conda create -n StreetFighterAI python=3.8.10
conda activate StreetFighterAI

# Install Python libraries
cd [parent_directory_of_project]/street-fighter-ai/main
pip install -r requirements.txt

# Run script to locate gym-retro game folder
cd ..
python .\utils\print_game_lib_folder.py
```

After the console outputs the folder path, copy it to the file explorer and navigate to the corresponding path. This folder contains the game data files for "Street Fighter II: Special Champion Edition" within gym-retro, including the game ROM file and data configuration files. Copy the `Champion.Level12.RyuVsBison.state`, `data.json`, `metadata.json`, and `scenario.json` files from the `data/` folder of this project into the game data folder, replacing the original files (administrator privileges may be required). The `.state` file is a save state for the game's highest difficulty level, while the three `.json` files are gym-retro configuration files storing game information memory addresses (this project only uses [agent_hp] and [enemy_hp] for reading character health values in real-time).

To run the program, you will also need the game ROM file for "Street Fighter II: Special Champion Edition", which is not provided by gym-retro and must be obtained legally through other means. You can refer to this [link](https://wowroms.com/en/roms/sega-genesis-megadrive/street-fighter-ii-special-champion-edition-europe/26496.html).

Once you have legally obtained the game ROM file, copy it to the aforementioned gym-retro game data folder and rename it to `rom.md`. At this point, the environment setup is complete.

Note 1: If you want to manually capture save states and find memory variables in the game, you can use the gym-retro integration ui. Copy `data/Gym Retro Integration.exe` to the parent menu (two levels up, `retro/` folder) of the aforementioned gym-retro game data folder.

Note 2: If you want to record videos of the AI agent's gameplay, you will need to install [ffmpeg](https://ffmpeg.org/).

```bash
conda install ffmpeg
```

### <a name="running-tests"></a>Running Tests

Once the environment is set up, you can run `test.py` in the `main/` folder to test and experience the AI agent's performance at different stages of training.

```bash
cd [parent_directory_of_project]/street-fighter-ai/main
python test.py
```

Model weight files are stored in the `main/trained_models/` folder. The default model used in `test.py` is `ppo_ryu_2500000_steps_updated.zip`, which has good generalization and is capable of beating the final level of Street Fighter II: Special Champion Edition. If you want to see the performance of other models, you can change the `model_path` variable in `test.py` to the path of another model file. The observed performance of the models at various training stages is as follows:

* ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
* ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
* ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
* ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

### Training the Model

If you want to train your own model, you can run `train.py` in the `main/` folder.

```bash
cd [parent_directory_of_project]/street-fighter-ai/main
python train.py
```

### Viewing Training Curves

The project includes Tensorboard graphs of the training process. You can use Tensorboard to view detailed data. It is recommended to use the integrated Tensorboard plugin in VSCode to view the data directly. The traditional viewing method is listed below:

```bash
cd [parent_directory_of_project]/street-fighter-ai/main
tensorboard --logdir=logs/
```

Open the default Tensorboard service address `http://localhost:6006/` in your browser to view interactive graphs of the training process.

## Acknowledgements
This project uses open-source libraries such as [OpenAI Gym Retro](https://retro.readthedocs.io/en/latest/getting_started.html), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The contributions of all the developers to the open-source community are appreciated!

Two papers that had a significant impact on this project:

[1] [DIAMBRA Arena A New Reinforcement Learning Platform for Research and Experimentation](https://arxiv.org/abs/2210.10595)
The valuable summary of the experience in setting hyperparameters for deep reinforcement learning models in fighting games in this paper was of great help to the training process of this project.

[2] [Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf)
The "penalty decay" mechanism proposed in this paper effectively solved the "cowardice" problem (always avoiding opponents and not daring to even try attacking moves).