import time 

import retro
from stable_baselines3 import PPO

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
state_stages = [
    "Champion.Level1.RyuVsGuile",
    "Champion.Level1.ChunLiVsGuile", # Average reward for random strategy: -102.3 | -20.4
    "ChampionX.Level1.ChunLiVsKen", # Average reward for random strategy: -247.6
    "Champion.Level2.ChunLiVsKen",
    "Champion.Level3.ChunLiVsChunLi",
    "Champion.Level4.ChunLiVsZangief",
    "Champion.Level5.ChunLiVsDhalsim",
    "Champion.Level6.ChunLiVsRyu",
    "Champion.Level7.ChunLiVsEHonda",
    "Champion.Level8.ChunLiVsBlanka",
    "Champion.Level9.ChunLiVsBalrog",
    "Champion.Level10.ChunLiVsVega",
    "Champion.Level11.ChunLiVsSagat",
    "Champion.Level12.ChunLiVsBison"
    # Add other stages as necessary
]

env = make_env(game, state_stages[0])()

model = PPO(
    "CnnPolicy", 
    env,
    verbose=1
)
model_path = r"trained_models_ryu_level_1_reward_x3/ppo_ryu_6600000_steps"
model.load(model_path)
# Average reward for optuna/trial_1_best_model: -82.3
# Average reward for optuna/trial_9_best_model: 36.7 | -86.23
# Average reward for trained_models/ppo_chunli_5376000_steps: -77.8


obs = env.reset()
done = False

num_episodes = 30
episode_reward_sum = 0
for _ in range(num_episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while True:
        timestamp = time.time()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {}, playerHP: {}, enemyHP:{}".format(reward, info['health'], info['enemy_health']))
        env.render()
        # time.sleep(0.005)
    # print("Total reward: {}".format(total_reward))
    # episode_reward_sum += total_reward

# env.close()
# print("Average reward for {}: {}".format(model_path, episode_reward_sum/num_episodes))