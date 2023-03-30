import gym
import retro
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from custom_cnn import CustomCNN
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

def make_env(game, state, seed=0):
    def _init():
        env = retro.RetroEnv(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def objective(trial):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = make_env(game, state="ChampionX.Level1.ChunLiVsKen")()

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 256, 8192, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)

    # Using CustomCNN as the feature extractor
    policy_kwargs = {
        'features_extractor_class': CustomCNN
    }

    # Train the model
    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    for iteration in range(10):
        model.learn(total_timesteps=100000)
        mean_reward, _std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        trial.report(mean_reward, iteration)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=7200)  # Run optimization for 100 trials or 2 hours, whichever comes first

print("Best trial:")
trial = study.best_trial

print(" Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
