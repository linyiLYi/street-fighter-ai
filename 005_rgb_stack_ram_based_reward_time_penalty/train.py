import os
import random

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

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
    
# class StageIncreaseCallback(BaseCallback):
#     def __init__(self, stages, stage_interval, save_dir, verbose=0):
#         super(StageIncreaseCallback, self).__init__(verbose)
#         self.stages = stages
#         self.stage_interval = stage_interval
#         self.save_dir = save_dir
#         self.current_stage = 0

#     def _on_step(self) -> bool:
#         if self.n_calls % self.stage_interval == 0 and self.current_stage < len(self.stages) - 1:
#             self.current_stage += 1
#             new_state = self.stages[self.current_stage]
#             self.training_env.env_method("load_state", new_state, indices=None)
#             self.model.save(os.path.join(self.save_dir, f"ppo_chunli_stage_{self.current_stage}.zip"))
#         return True
    
def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"

    state_stages = [
        "Champion.Level1.RyuVsGuile",
        "Champion.Level1.ChunLiVsGuile", # Average reward for random strategy: -102.3
        "Champion.Level2.ChunLiVsKen",
        "Champion.Level3.ChunLiVsChunLi",
        "Champion.Level4.ChunLiVsZangief",
        "Champion.Level5.ChunLiVsDhalsim",
        "Champion.Level6.ChunLiVsRyu",
        "Champion.Level7.ChunLiVsEHonda",
        "Champion.Level8.ChunLiVsBlanka",
        "Champion.Level9.ChunLiVsBalrog",
        "Champion.Level10.ChunLiVsVega",
        "Champion.Level11.ChunLiVsSagat",
        "Champion.Level12.ChunLiVsBison"
        # Add other stages as necessary
    ]

    # state_stages = [
    #     "ChampionX.Level1.ChunLiVsKen", # Average reward for random strategy: -247.6
    #     "ChampionX.Level2.ChunLiVsChunLi",
    #     "ChampionX.Level3.ChunLiVsZangief",
    #     "ChampionX.Level4.ChunLiVsDhalsim",
    #     "ChampionX.Level5.ChunLiVsRyu",
    #     "ChampionX.Level6.ChunLiVsEHonda",
    #     "ChampionX.Level7.ChunLiVsBlanka",
    #     "ChampionX.Level8.ChunLiVsGuile",
    #     "ChampionX.Level9.ChunLiVsBalrog",
    #     "ChampionX.Level10.ChunLiVsVega",
    #     "ChampionX.Level11.ChunLiVsSagat",
    #     "ChampionX.Level12.ChunLiVsBison"
    #     # Add other stages as necessary
    # ]
    # Champion is at difficulty level 4, ChampionX is at difficulty level 8.

    env = make_env(game, state_stages[0])()

    # Warp env in Monitor wrapper to record training progress
    env = Monitor(env, LOG_DIR)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        tensorboard_log="logs"
    )

    # Set the save directory
    save_dir = "trained_models_ryu_level_1_time_reward"
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models/ppo_chunli_1296000_steps.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": 0.0002
    # }
    # model = PPO.load(model_path, env=env, device="cuda")#, custom_objects=custom_objects)

    # Set up callbacks
    # opponent_interval = 35840 # stage_interval * num_envs = total_steps_per_stage
    checkpoint_interval = 200000 # checkpoint_interval * num_envs = total_steps_per_checkpoint (Every 80 rounds)
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_ryu")
    # stage_increase_callback = RandomOpponentChangeCallback(state_stages, opponent_interval, save_dir)

    # model_params = {
    #     'n_steps': 5, 
    #     'gamma': 0.99, 
    #     'gae_lambda':1, 
    #     'learning_rate': 7e-4, 
    #     'vf_coef': 0.5,
    #     'ent_coef': 0.0,
    #     'max_grad_norm':0.5,
    #     'rms_prop_eps':1e-05 
    # }
    # model = A2C('CnnPolicy', env, tensorboard_log='logs/', verbose=1, **model_params, policy_kwargs=dict(optimizer_class=RMSpropTF))

    model.learn(
        total_timesteps=int(10000000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
        callback=[checkpoint_callback]#, stage_increase_callback]
    )
    env.close()

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_sf2_ryu_final.zip"))

if __name__ == "__main__":
    main()
