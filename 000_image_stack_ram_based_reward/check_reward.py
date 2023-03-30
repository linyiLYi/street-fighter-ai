import time 

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from street_fighter_custom_wrapper import StreetFighterCustomWrapper
    
def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
state = "Champion.Level1.ChunLiVsGuile"#"ChampionX.Level1.ChunLiVsKen"

env = make_env(game, state)()
env = Monitor(env, 'logs/')

num_episodes = 30
episode_reward_sum = 0
for _ in range(num_episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        timestamp = time.time()
        obs, reward, done, info = env.step(env.action_space.sample())

        if reward != 0:
            total_reward += reward
            print("Reward: {}, playerHP: {}, enemyHP:{}".format(reward, info['health'], info['enemy_health']))
        env.render()
    print("Total reward: {}".format(total_reward))
    episode_reward_sum += total_reward

env.close()
print("Average reward for random strategy: {}".format(episode_reward_sum/num_episodes))
