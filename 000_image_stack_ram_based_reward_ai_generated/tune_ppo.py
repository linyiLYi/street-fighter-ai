import os

import retro
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

LOG_DIR = 'logs/'
OPT_DIR = 'optuna/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 1024, 8192, log=True),
        'gamma':trial.suggest_float('gamma', 0.9, 0.9999),
        'learning_rate':trial.suggest_float('learning_rate', 5e-5, 1e-4, log=True),
        'clip_range':trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_float('gae_lambda', 0.8, 0.99)
    }

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

def optimize_agent(trial):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level1.ChunLiVsGuile"#"ChampionX.Level1.ChunLiVsKen"

    try:
        model_params = optimize_ppo(trial) 

        # Create environment 
        env = make_env(game, state)()
        env = Monitor(env, LOG_DIR)

        # Create algo 
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
        model.learn(total_timesteps=100000)

        # Evaluate model 
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=30)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward
    
    except Exception as e:
        return -1
    
# Creating the experiment 
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=10, n_jobs=1)

print(study.best_params)
print(study.best_trial)
