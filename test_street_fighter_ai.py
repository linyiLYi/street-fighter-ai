import time 

import torch
import gym
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

rom_directory = "C:/Users/unitec/Documents/AIProjects/street-fighter-ai"
retro.data.Integrations.add_custom_path(rom_directory)

env = retro.RetroEnv(
    game='StreetFighterIISpecialChampionEdition-Genesis', 
    state='Champion.Level3.ChunLiVsChunLi'
)
# Champion.Level2.ChunLiVsKen
# Champion.Level3.ChunLiVsChunLi


env = DummyVecEnv([lambda: env])

model = PPO("CnnPolicy", env)
model.load("trained_models/ppo_sf2_chunli_final")

obs = env.reset()
while True:
    timestamp = time.time()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    render_time = time.time() - timestamp
    if render_time < 0.0111:
        time.sleep(0.0111 - render_time)  # Add a delay for 90 FPS
    if done:
        break
        obs = env.reset()

env.close()