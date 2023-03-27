import gym

# Create a custom environment for Street Fighter II
class CustomStreetFighterEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomStreetFighterEnv, self).__init__(env)
        self.previous_health = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        # Reward function
        custom_reward = self.custom_reward_function(info)

        return observation, custom_reward, done, info
    
    def reset(self):
        self.previous_health = 0
        return self.env.reset()

    def custom_reward_function(self, info):
        # Reward weights
        health_weight = 1
        hit_weight = 2
        block_weight = 1
        knockdown_weight = 5

        # Retrieve relevant information from info
        player_health = info["health1"]
        opponent_health = info["health2"]
        player_is_hit = info["is_hit1"]
        opponent_is_hit = info["is_hit2"]
        player_is_blocking = info["is_blocking1"]
        # opponent_is_blocking = info["is_blocking2"]
        player_is_knockdown = info["is_knockdown1"]
        opponent_is_knockdown = info["is_knockdown2"]

        # Compute reward components
        health_reward = (player_health - opponent_health) * health_weight
        hit_reward = hit_weight if opponent_is_hit else 0
        block_reward = block_weight if player_is_blocking else 0
        knockdown_reward = knockdown_weight if opponent_is_knockdown else 0

        # Penalty components
        hit_penalty = -hit_weight if player_is_hit else 0
        knockdown_penalty = -knockdown_weight if player_is_knockdown else 0

        # Calculate total custom reward
        custom_reward = health_reward + hit_reward + block_reward + knockdown_reward + hit_penalty + knockdown_penalty

        return custom_reward