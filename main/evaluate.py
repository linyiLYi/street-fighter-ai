# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import retro

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

RESET_ROUND = True # Reset the round when fight is over. 
RENDERING = False
MODEL_PATH = r"trained_models/ppo_ryu_2000000_steps"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        env = Monitor(env)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsBison")()
model = PPO("CnnPolicy", env)
model.load(MODEL_PATH)
mean_reward, std_reward = evaluate_policy(model, env, render=False, n_eval_episodes=5, deterministic=False, return_episode_rewards=True)
print(mean_reward)
print(std_reward)
# print(f"Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
