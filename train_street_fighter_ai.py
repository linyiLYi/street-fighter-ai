import os

import gym
import torch
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

def import_rom(rom_directory):
    retro.data.Integrations.add_custom_path(rom_directory)
    os.system(f'python -m retro.import "{rom_directory}"')

class StageIncreaseCallback(BaseCallback):
    def __init__(self, stages, stage_interval, save_dir, verbose=0):
        super(StageIncreaseCallback, self).__init__(verbose)
        self.stages = stages
        self.stage_interval = stage_interval
        self.save_dir = save_dir
        self.current_stage = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.stage_interval == 0 and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            new_state = self.stages[self.current_stage]
            self.training_env.env_method("load_state", new_state, indices=None)
            self.model.save(os.path.join(self.save_dir, f"ppo_chunli_stage_{self.current_stage}.zip"))
        return True

def make_env(game, state, seed=0):
    def _init():
        env = retro.RetroEnv(game=game, state=state)
        env.seed(seed)
        return env
    return _init

def main():
    n_envs = 4

    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state_stages = [
        "Champion.Level1.ChunLiVsGuile",
        "Champion.Level2.ChunLiVsKen",
        "Champion.Level3.ChunLiVsChunLi",
        "Champion.Level4.ChunLiVsZangief",
        # Add other stages as necessary
    ]

    rom_directory = "C:/Users/unitec/Documents/AIProjects/street-fighter-ai"
    import_rom(rom_directory)

    # Create the environment with the correct game ID and scenario
    # env = retro.RetroEnv(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.ChunLiVsGuile')
    env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(n_envs)])

    # env = DummyVecEnv([lambda: env])

    # Define PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=2.5e-4,
        n_steps=5400,
        batch_size=96,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=None,
    )

    # Set the save directory
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Set up callbacks
    stage_interval = 540000 # Number of steps between increasing stages
    checkpoint_callback = CheckpointCallback(save_freq=stage_interval, save_path=save_dir, name_prefix="ppo_chunli")
    stage_increase_callback = StageIncreaseCallback(state_stages, stage_interval, save_dir)

    # Train the model and save intermediate models
    model.learn(total_timesteps=1620000, callback=[checkpoint_callback, stage_increase_callback])

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_sf2_chunli_final.zip"))

    env.close()

if __name__ == "__main__":
    main()
    