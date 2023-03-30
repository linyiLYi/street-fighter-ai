import gym
import cv2
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, testing=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env
        self.testing = testing
        
        # Store the previous frame
        self.prev_frame = None

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        # Update observation space to include one grayscale frame difference image
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    
    def _preprocess_observation(self, observation):
        obs_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        obs_gray_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        return obs_gray_resized

    def reset(self):
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        
        observation = self.env.reset()
        # Reset the previous frame
        self.prev_frame = self._preprocess_observation(observation)
        return np.zeros_like(self.prev_frame)

    def step(self, action):
        observation, _reward, _done, info = self.env.step(action)

        obs_gray_resized = self._preprocess_observation(observation)

        if self.prev_frame is not None:
            frame_delta = obs_gray_resized - self.prev_frame
        else:
            frame_delta = np.zeros_like(obs_gray_resized)

        self.prev_frame = obs_gray_resized

        # During fighting, either player or opponent has positive health points.
        if info['health'] > 0 or info['enemy_health'] > 0:

            # Player Loses
            if info['health'] < 0 and info['enemy_health'] > 0:
                reward = (-self.full_hp) * info['enemy_health']
                done = True

            # Player Wins
            elif info['enemy_health'] < 0 and info['health'] > 0:
                reward = self.full_hp * info['health']
                done = True

            # During Fighting
            else:
                reward = (self.prev_oppont_health - info['enemy_health']) - (self.prev_player_health - info['health'])

        self.prev_player_health = info['health']
        self.prev_oppont_health = info['enemy_health']

        if self.testing:
            done = False
             
        return frame_delta, reward, done, info
    