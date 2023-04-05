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

env = make_env(game, state="Champion.Level12.RyuVsBison")()

model = PPO(
    "CnnPolicy", 
    env,
    verbose=1
)
model_path = r"trained_models_ryu_vs_bison_finetune/ppo_ryu_9500000_steps.zip"
model.load(model_path)

# obs = env.reset()
done = False

num_episodes = 100
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
        # time.sleep(0.002)
    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1
    print("Total reward: {}".format(total_reward))
    episode_reward_sum += total_reward

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
print("Average reward for {}: {}".format(model_path, episode_reward_sum/num_episodes))