import os
import time 

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

RESET_ROUND = False # Reset the round when fight is over. 
RENDERING = True
RECORDING = True

RANDOM_ACTION = False
MODEL_DIR = r"trained_models/"
MOVIE_DIR = r"recordings"

MODEL_NAME = r"ppo_ryu_7000000_steps"

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
# model = PPO("CnnPolicy", env)

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

# obs = env.reset()
done = False

num_episodes = 30
episode_reward_sum = 0
num_victory = 0
for _ in range(num_episodes):
    done = False
    obs = env.reset()

    if RECORDING:
        # Start recording
        movie_path = os.path.join(MOVIE_DIR, "{}.bk2".format(MODEL_NAME))
        env.unwrapped.movie = retro.Movie(movie_path, retro.MovieMode.RECORD)
        env.unwrapped.movie.step()

    total_reward = 0
    
    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
        
        if RECORDING:
            # Record the step
            env.unwrapped.movie.step()

        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
    
    if RECORDING:
        # Stop recording
        env.unwrapped.movie.close()
        del env.unwrapped.movie

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1
    print("Total reward: {}".format(total_reward))
    episode_reward_sum += total_reward

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))