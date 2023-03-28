import time 

import cv2
import torch
import gym
import retro
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def check_done(screen, win_template, lose_template, threshold=0.65):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    win_res = cv2.matchTemplate(gray_screen, win_template, cv2.TM_CCOEFF_NORMED)
    lose_res = cv2.matchTemplate(gray_screen, lose_template, cv2.TM_CCOEFF_NORMED)
    
    if np.max(win_res) >= threshold:
        print("You win!")
        return True
    
    if np.max(lose_res) >= threshold:
        print("You lose!")
        return True

def get_health_points(screen):
    # Get the player's HP
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    player_health_area = gray_screen[15:20, 32:120]
    oppoent_health_area = gray_screen[15:20, 136:224]
    
    # Get health points using the number of pixels above 129.
    player_health = np.sum(player_health_area > 129) / player_health_area.size
    oppoent_health = np.sum(oppoent_health_area > 129) / oppoent_health_area.size

    # Helper function to get the max and min pixel values.
    # max_pixel = np.max(player_health_area)
    # min_pixel = np.min(player_health_area)
    # avg = (max_pixel + min_pixel) / 2

    return player_health, oppoent_health
    
rom_directory = "C:/Users/unitec/Documents/AIProjects/street-fighter-ai"
retro.data.Integrations.add_custom_path(rom_directory)

env = retro.RetroEnv(
    game='StreetFighterIISpecialChampionEdition-Genesis', 
    state='Champion.Level3.ChunLiVsChunLi'
)
# Champion.Level1.ChunLiVsGuile
# Champion.Level2.ChunLiVsKen
# Champion.Level3.ChunLiVsChunLi

# env = DummyVecEnv([lambda: env])

model = PPO("CnnPolicy", env)
model.load("trained_models/ppo_sf2_chunli_final")

obs = env.reset()
game_over = False

win_template = cv2.imread('images/pattern_wins_gray.png', cv2.IMREAD_GRAYSCALE)
lose_template = cv2.imread('images/pattern_lose_gray.png', cv2.IMREAD_GRAYSCALE)

while not game_over:
    timestamp = time.time()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    screen = env.unwrapped.get_screen()
    get_health_points(screen)
    game_over = check_done(screen, win_template, lose_template)
    render_time = time.time() - timestamp
    if render_time < 0.0111:
        time.sleep(0.0111 - render_time)  # Add a delay for 90 FPS

env.close()