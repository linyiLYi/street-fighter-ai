import os
import sys

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        # Create log directory
        env_log_dir = os.path.join(LOG_DIR, str(seed+100)) # +100 to avoid conflict with other log dirs when fine-tuning
        os.makedirs(env_log_dir, exist_ok=True)
        env = Monitor(env, env_log_dir)
        env.seed(seed)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state="Champion.Level12.RyuVsBison", seed=i) for i in range(NUM_ENV)])

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )

    # Set the save directory
    save_dir = "trained_models_ryu_vs_bison_no_coef_no_time_reward_9_frame"
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models_ryu_vs_bison_finetune/ppo_ryu_10000000_steps.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set up callbacks
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_ryu")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(100000000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback]#, stage_increase_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_sf2_ryu_final.zip"))

if __name__ == "__main__":
    main()
