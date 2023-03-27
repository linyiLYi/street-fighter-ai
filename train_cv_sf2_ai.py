import gym
import cv2
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# Custom feature extractor (CNN)
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(CustomCNN, self).__init__(observation_space, features_dim=512)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, self.features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

# Custom environment wrapper for preprocessing
class CustomAtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def _preprocess_observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(observation, axis=-1)

    def reset(self):
        observation = self.env.reset()
        return self._preprocess_observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._preprocess_observation(observation), reward, done, info

def make_env(game, state, seed=0):
    def _init():
        env = retro.RetroEnv(game=game, state=state, obs_type=retro.Observations.IMAGE)
        env = CustomAtariWrapper(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state_stages = [
        "Champion.Level1.ChunLiVsGuile",
        "Champion.Level2.ChunLiVsKen",
        "Champion.Level3.ChunLiVsChunLi",
        "Champion.Level4.ChunLiVsZangief",
        # Add other stages as necessary
    ]

    num_envs = 8
    seed = 42

    env = SubprocVecEnv([make_env(game, state_stages[0], seed=i) for i in range(num_envs)])

    policy_kwargs = {
    'features_extractor_class': CustomCNN
    }

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        policy_kwargs=policy_kwargs, 
        verbose=1
    )
    model.learn(total_timesteps=int(1000))

    model.save("ppo_sf2_cnn")

if __name__ == "__main__":
    main()

# missing reward function