import time 

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_cnn import CustomCNN
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
    
def make_env(game, state):
    def _init():
        env = retro.RetroEnv(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, testing=True)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
state = "Champion.Level1.ChunLiVsGuile"

env = make_env(game, state)()
model = PPO.load(r"trained_models_continued/ppo_chunli_6048000_steps")
obs = env.reset()
done = False

while not done:
    timestamp = time.time()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(info)
    if reward != 0:
        print(reward, info['health'], info['enemy_health'])
    env.render()

env.close()