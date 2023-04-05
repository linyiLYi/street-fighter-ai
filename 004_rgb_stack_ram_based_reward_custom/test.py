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
    "Champion.Level2.RyuVsKen",
    "Champion.Level3.RyuVsChunLi",
    "Champion.Level4.RyuVsZangief",
    "Champion.Level5.RyuVsDhalsim",
    "Champion.Level6.RyuVsRyu",
    "Champion.Level7.RyuVsEHonda",
    "Champion.Level8.RyuVsBlanka",
    "Champion.Level9.RyuVsBalrog",
    "Champion.Level10.RyuVsVega",
    "Champion.Level11.RyuVsSagat",
    "Champion.Level12.RyuVsBison"
]
# state_stages = [
#     "Champion.Level1.RyuVsGuile",
#     "Champion.Level1.ChunLiVsGuile", # Average reward for random strategy: -102.3 | -20.4
#     "ChampionX.Level1.ChunLiVsKen", # Average reward for random strategy: -247.6
#     "Champion.Level2.ChunLiVsKen",
#     "Champion.Level3.ChunLiVsChunLi",
#     "Champion.Level4.ChunLiVsZangief",
#     "Champion.Level5.ChunLiVsDhalsim",
#     "Champion.Level6.ChunLiVsRyu",
#     "Champion.Level7.ChunLiVsEHonda",
#     "Champion.Level8.ChunLiVsBlanka",
#     "Champion.Level9.ChunLiVsBalrog",
#     "Champion.Level10.ChunLiVsVega",
#     "Champion.Level11.ChunLiVsSagat",
#     "Champion.Level12.ChunLiVsBison"
#     # Add other stages as necessary
# ]

env = make_env(game, state_stages[0])()

model = PPO(
    "CnnPolicy", 
    env,
    verbose=1
)
model_path = r"trained_models_ryu_level_1_time_reward_small_loop_continue/ppo_ryu_5000000_steps.zip"
model.load(model_path)
# Average reward for optuna/trial_1_best_model: -82.3
# Average reward for optuna/trial_9_best_model: 36.7 | -86.23
# Average reward for trained_models/ppo_chunli_5376000_steps: -77.8

# Level_1 Average reward for trained_models_ryu_level_1_time_reward_small_random/ppo_ryu_4200000_steps: 0.35772262101207986 Winning rate: 0.5666666666666667
# Level_2 Average reward for trained_models_ryu_level_1_time_reward_small_random/ppo_ryu_4200000_steps: 0.18094390738868166 Winning rate: 0.16666666666666666

# obs = env.reset()
done = False

num_episodes = 12
episode_reward_sum = 0
num_victory = 0
for _ in range(num_episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
    # while True:
        timestamp = time.time()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        env.render()
        # time.sleep(0.005)
    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1
    print("Total reward: {}".format(total_reward))
    episode_reward_sum += total_reward

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
print("Average reward for {}: {}".format(model_path, episode_reward_sum/num_episodes))