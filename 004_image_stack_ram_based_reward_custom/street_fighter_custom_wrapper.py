import math
import collections

import gym
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, testing=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 4 frames
        self.num_frames = 3
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        # Update observation space to include stacked grayscale images
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.testing = testing
    
    def _preprocess_observation(self, observation):

        # Stack the downsampled frames.
        self.frame_stack.append(observation[::2, ::2, :])

        # Stack the R, G, B channel of each frame and return the "image".
        stacked_image = np.stack([frame[:, :, i] for i, frame in enumerate(self.frame_stack)], axis=-1)
        return stacked_image

    def reset(self):
        observation = self.env.reset()
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([frame[:, :, i] for i, frame in enumerate(self.frame_stack)], axis=-1)

    def step(self, action):
        
        obs, _reward, _done, info = self.env.step(action)
        curr_player_health = info['health']
        curr_oppont_health = info['enemy_health']
        
        self.total_timesteps += 1

        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                   # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # During testing, the session should always keep going.
        if self.testing:
            custom_done = False
             
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) 
        return self._preprocess_observation(obs), 0.001 * custom_reward, custom_done, info # reward normalization
    