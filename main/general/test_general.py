# Copyright 2024 WANG Jing. All Rights Reserved.
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

import os
import time 

import retro
from stable_baselines3 import PPO
from street_fighter_custom_wrapper_general import StreetFighterCustomWrapper

RESET_ROUND = False  # Already in rule best of three keep in False 
RENDERING = True    # Whether to render the game screen.

lin = r"ppo_ryu_2500000_steps_updated"
king = r"ppo_ryu_king_10000000_steps" 

#the trained result performance in best of three games
# 2500000_steps_updated which with 1 round trained got win rate in 0.4
# king_10000000_steps which with entire best of three rules got win rate in 0.99

MODEL_NAME = king 
RANDOM_ACTION = False
NUM_EPISODES = 50 # 
MODEL_DIR = r"./trained_models"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsBison")()

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False 
    total_reward = 0
    while not done:
        timestamp = time.time()
        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        
    if info['enemy_hp'] < 0 and info['agent_hp']>=0:
        print("Victory!")
        num_victory += 1
    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward
    obs = env.reset()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))