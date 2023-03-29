import os
import random

import gym
import cv2
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from cnn_lstm import CNNLSTM, CNNEncoder
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

class RandomOpponentChangeCallback(BaseCallback):
    def __init__(self, stages, opponent_interval, verbose=0):
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
        env = retro.RetroEnv(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        env.seed(seed)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state_stages = [
        "ChampionX.Level1.ChunLiVsKen",
        "ChampionX.Level2.ChunLiVsChunLi",
        "ChampionX.Level3.ChunLiVsZangief",
        "ChampionX.Level4.ChunLiVsDhalsim",
        "ChampionX.Level5.ChunLiVsRyu",
        "ChampionX.Level6.ChunLiVsEHonda",
        "ChampionX.Level7.ChunLiVsBlanka",
        "ChampionX.Level8.ChunLiVsGuile",
        "ChampionX.Level9.ChunLiVsBalrog",
        "ChampionX.Level10.ChunLiVsVega",
        "ChampionX.Level11.ChunLiVsSagat",
        "ChampionX.Level12.ChunLiVsBison"
        # Add other stages as necessary
    ]
    # Champion is at difficulty level 4, ChampionX is at difficulty level 8.

    num_envs = 8

    # env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(num_envs)])
    env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(num_envs)])

    class CustomPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs)

            self.features_extractor = CNNLSTM()

    model = PPO(
        CustomPolicy, 
        env,
        device="cuda",  
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
    save_dir = "trained_models"
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
