import os
import random

import gym
import cv2
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch
import torch.nn as nn

from custom_cnn import CustomCNN
from custom_sf2_cv_env import StreetFighterCustomWrapper

class RandomOpponentChangeCallback(BaseCallback):
    def __init__(self, stages, opponent_interval, save_dir, verbose=0):
        super(RandomOpponentChangeCallback, self).__init__(verbose)
        self.stages = stages
        self.opponent_interval = opponent_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.opponent_interval == 0:
            new_state = random.choice(self.stages)
            print("\nCurrent state:", new_state)
            self.training_env.env_method("load_state", new_state, indices=None)
        return True
    
def make_env(game, state, seed=0):
    def _init():
        win_template = cv2.imread('images/pattern_win_gray.png', cv2.IMREAD_GRAYSCALE)
        lose_template = cv2.imread('images/pattern_lose_gray.png', cv2.IMREAD_GRAYSCALE)
        env = retro.RetroEnv(game=game, state=state, obs_type=retro.Observations.IMAGE)
        env = StreetFighterCustomWrapper(env, win_template, lose_template)
        # env.seed(seed)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state_stages = [
        # "Champion.Level1.ChunLiVsGuile",
        # "Champion.Level2.ChunLiVsKen",
        # "Champion.Level3.ChunLiVsChunLi",
        # "Champion.Level4.ChunLiVsZangief",
        # "Champion.Level5.ChunLiVsDhalsim",
        "Champion.Level6.ChunLiVsRyu",
        "Champion.Level7.ChunLiVsEHonda",
        "Champion.Level8.ChunLiVsBlanka",
        "Champion.Level9.ChunLiVsBalrog",
        "Champion.Level10.ChunLiVsVega",
        "Champion.Level11.ChunLiVsSagat",
        "Champion.Level12.ChunLiVsBison"
        # Add other stages as necessary
    ]

    num_envs = 8

    # env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(num_envs)])
    env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(num_envs)])

    policy_kwargs = {
        'features_extractor_class': CustomCNN
    }

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        n_steps=5400,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        ent_coef=0.01,
        clip_range=0.2,
        clip_range_vf=None,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1
    )

    # Set the save directory
    save_dir = "trained_models_cv_level6up"
    os.makedirs(save_dir, exist_ok=True)

    # Set up callbacks
    opponent_interval = 5400 # stage_interval * num_envs = total_steps_per_stage
    checkpoint_interval = 54000 # checkpoint_interval * num_envs = total_steps_per_checkpoint (Every 80 rounds)
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_chunli")
    stage_increase_callback = RandomOpponentChangeCallback(state_stages, opponent_interval, save_dir)

    
    model.learn(
        total_timesteps=int(6048000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
        callback=[checkpoint_callback, stage_increase_callback]
    )

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_sf2_chunli_final.zip"))

if __name__ == "__main__":
    main()
