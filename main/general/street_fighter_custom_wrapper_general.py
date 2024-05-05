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

import math
import time
import collections

import gym
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):

    def __init__(self, env, reset_round=False, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        self.win = 0
        self.loss = 0
        self.round = 0
        self.round_end = False
        self.jump = False

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0
        self.match1_reward = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
        
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        self.win = 0
        self.loss = 0
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
    

    def step(self, action):
        custom_done = False
                
        obs, _reward, _done, info = self.env.step(action)
        if not self.jump:
            self.frame_stack.append(obs[::2, ::2, :])
        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            #time.sleep(0.003)     
        for _ in range(self.num_step_frames - 1):           
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info= self.env.step(action)
            if not self.jump:
                self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                #time.sleep(0.003)
                
        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp'] 

        if not custom_done:        
            if self.round_end:
                while not(curr_player_health == self.full_hp and curr_oppont_health == self.full_hp):
                    self.jump = True
                    obs, _reward, _done, info = self.env.step([0] * 12)
                    curr_player_health = info['agent_hp']
                    curr_oppont_health = info['enemy_hp'] 
                    if self.rendering:
                        self.env.render()
                                #time.sleep(0.01)  
                self.round_end = False
                self.jump = False         
        if self.jump :
            custom_reward = np.nan
            self.prev_player_health = self.full_hp
            self.prev_oppont_health = self.full_hp
            custom_done = False  
        else :           
            self.total_timesteps += self.num_step_frames
            reduce_enermy_health = self.prev_oppont_health - curr_oppont_health
            reduce_player_health = self.prev_player_health - curr_player_health 
     
            # Determine game status and calculate rewards.
            if reduce_player_health > 0 and curr_player_health < 0:
                custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))
                self.loss += 1
                self.round +=1
                self.round_end = True  
                #print('loss')                 
                if self.loss == 2:
                    self.loss = 0
                    self.round = 0
                    custom_done = True
            
            elif reduce_enermy_health > 0 and curr_oppont_health <0 :
                custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
                self.win += 1
                self.round +=1
                self.round_end = True
                #print('win')
                if self.win == 2:
                    self.win = 0
                    self.round = 0
                    custom_done = True            
            else:
                if reduce_enermy_health >= 0 and reduce_player_health >= 0:
                    if reduce_enermy_health > 0 or reduce_player_health > 0:
                        custom_reward = self.reward_coeff * (reduce_enermy_health - reduce_player_health)
                    else:
                        custom_reward = 0
                else:
                    custom_reward = 0

            # Update health states.
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health

        
        # Max reward is 2 * 6 * full_hp = 2108  2 * (damage * 3 + winning_reward * 3) norm_coefficient = 0.001 for win first two
        # Or in three round 2 * 6 * full_hp - full_hp - 1 + 3 * full_hp = 2464 (damage * 3 + winning_reward * 3) 
        #print("reward{}, reduce_player_health:{}, reduce_enermy_health:{}, agentHP:{}, enemyHP:{}".format(custom_reward,reduce_player_health,reduce_enermy_health,curr_player_health,curr_oppont_health))  
        return self._stack_observation(), 0.001 * custom_reward, custom_done, info # reward normalization
 
 
