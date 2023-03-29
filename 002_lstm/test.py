import time 

import cv2
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cnn_lstm import CNNLSTM, CNNEncoder
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
    
def make_env(game, state):
    def _init():
        win_template = cv2.imread('images/pattern_win_gray.png', cv2.IMREAD_GRAYSCALE)
        lose_template = cv2.imread('images/pattern_lose_gray.png', cv2.IMREAD_GRAYSCALE)
        env = retro.RetroEnv(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env, win_template, lose_template, testing=True)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
state_stages = [
    "Champion.Level1.ChunLiVsGuile",
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

# Wrap the environment
env = DummyVecEnv([lambda: env])

policy_kwargs = {
    'features_extractor_class': CNNEncoder,
    'net_arch': [512, 'lstm']
}

model = PPO(
    "CnnPolicy", 
    env,
    device="cuda", 
    policy_kwargs=policy_kwargs, 
    verbose=1
)
model.load(r"trained_models/ppo_chunli_1296000_steps")

obs = env.reset()
done = False

while True:
    timestamp = time.time()
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    render_time = time.time() - timestamp
    if render_time < 0.0111:
        time.sleep(0.0111 - render_time)  # Add a delay for 90 FPS

# env.close()
