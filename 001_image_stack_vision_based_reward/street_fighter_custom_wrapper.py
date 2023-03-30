import collections

import gym
import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize
from gym.spaces import MultiBinary

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, testing=False, threshold=0.65):
        super(StreetFighterCustomWrapper, self).__init__(env)
        
        # Use a deque to store the last 4 frames
        self.frame_stack = collections.deque(maxlen=4)

        self.threshold = threshold
        self.game_screen_gray = None

        self.prev_player_health = 1.0
        self.prev_opponent_health = 1.0

        # Update observation space to include 4 stacked grayscale images
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 84, 84), dtype=np.float32
        )

        self.testing = testing

        # Normalize the image for MobileNetV3Small.
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def _preprocess_observation(self, observation):
        self.game_screen_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(self.game_screen_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        # Add the resized image to the frame stack
        self.frame_stack.append(resized_image)

        # Stack the last 4 frames and return the stacked frames
        stacked_frames = np.stack(self.frame_stack, axis=0)
        return stacked_frames

    def _get_win_or_lose_bonus(self):
        if self.prev_player_health > self.prev_opponent_health:
            # print('You win!')
            return 300
        else:
            # print('You lose!')
            return -300
        
    def _get_reward(self):
        player_health_area = self.game_screen_gray[15:20, 32:120]
        oppoent_health_area = self.game_screen_gray[15:20, 136:224]
        
        # Get health points using the number of pixels above 129.
        player_health = np.sum(player_health_area > 129) / player_health_area.size
        opponent_health = np.sum(oppoent_health_area > 129) / oppoent_health_area.size

        player_health_diff = self.prev_player_health - player_health
        opponent_health_diff = self.prev_opponent_health - opponent_health

        reward = (opponent_health_diff - player_health_diff) * 200 # max would be 200

        # Penalty for each step without any change in health
        if opponent_health_diff <= 0.0000001:
            reward -= 12.0 / 60.0 # -12 points per second if no damage to opponent

        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health

        # Print the health values of the player and the opponent
        # print("Player health: %f Opponent health:%f" % (player_health, opponent_health))
        return reward

    def reset(self):
        observation = self.env.reset()
        self.prev_player_health = 1.0
        self.prev_opponent_health = 1.0
        
        # Clear the frame stack and add the first observation 4 times
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(self._preprocess_observation(observation)[0])

        return self._preprocess_observation(observation)

    def step(self, action):
        # observation, _, _, info = self.env.step(action)
        observation, _reward, _done, info = self.env.step(action)
        custom_reward = self._get_reward()
        custom_reward -= 1.0 / 60.0 # penalty for each step (-1 points per second)

        custom_done = False
        if self.prev_player_health <= 0.00001 or self.prev_opponent_health <= 0.00001:
            custom_reward += self._get_win_or_lose_bonus()
            if not self.testing:
                custom_done = True
            else:
                self.prev_player_health = 1.0
                self.prev_opponent_health = 1.0
             
        return self._preprocess_observation(observation), custom_reward, custom_done, info
    