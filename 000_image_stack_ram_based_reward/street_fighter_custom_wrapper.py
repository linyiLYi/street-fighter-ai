import collections

import gym
import cv2
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, testing=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 4 frames
        self.num_frames = 3
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        # Update observation space to include stacked grayscale images
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        self.testing = testing
    
    def _preprocess_observation(self, observation):
        obs_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        obs_gray_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add the resized image to the frame stack
        self.frame_stack.append(obs_gray_resized)

        # Stack the frames and return the "image"
        stacked_frames = np.stack(self.frame_stack, axis=-1)
        return stacked_frames

    def reset(self):
        observation = self.env.reset()
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        obs_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        obs_gray_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(obs_gray_resized)

        return np.stack(self.frame_stack, axis=-1)

    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        
        # During fighting, either player or opponent has positive health points.
        if info['health'] > 0 or info['enemy_health'] > 0:

            # Player Loses
            if info['health'] < 0 and info['enemy_health'] > 0:
                # reward = (-self.full_hp) * info['enemy_health'] * 0.05 # max = 0.05 * 176 * 176 = 1548.8
                reward = -info['enemy_health'] # Use the left over health points as penalty
                
                # Prevent data overflow
                if reward < -self.full_hp: 
                    reward = 0
                
                done = True

            # Player Wins
            elif info['enemy_health'] < 0 and info['health'] > 0:
                # reward = self.full_hp * info['health'] * 0.05
                reward = info['health']


                # Prevent data overflow
                if reward > self.full_hp:
                    reward = 0

                done = True

            # During Fighting
            else:
                reward = (self.prev_oppont_health - info['enemy_health']) - (self.prev_player_health - info['health'])

                # Prevent data overflow
                if reward > 99:
                    reward = 0

        self.prev_player_health = info['health']
        self.prev_oppont_health = info['enemy_health']

        if self.testing:
            done = False
             
        return self._preprocess_observation(obs), reward, done, info
    