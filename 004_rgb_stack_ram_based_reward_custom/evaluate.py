import retro

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from custom_cnn import CustomCNN
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
# env = Monitor(env, 'logs/')

policy_kwargs = {'features_extractor_class': CustomCNN}
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)

model = PPO.load(r"dummy_model_ppo_chunli")
# model.load(r"trained_models/ppo_chunli_864000_steps")

mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=10, deterministic=False, return_episode_rewards=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
