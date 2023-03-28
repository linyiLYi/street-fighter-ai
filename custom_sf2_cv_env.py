import gym
import cv2
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, win_template, lose_template, testing=False, threshold=0.65):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.win_template = win_template
        self.lose_template = lose_template
        self.threshold = threshold
        self.game_screen_gray = None

        self.prev_player_health = 1.0
        self.prev_opponent_health = 1.0

        # Update observation space to single-channel grayscale image
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84, 1), dtype=np.float32
        )

        self.testing = testing
    
    def _preprocess_observation(self, observation):
        self.game_screen_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(self.game_screen_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        return np.expand_dims(resized_image, axis=-1)

    def _get_win_or_lose_bonus(self):
        if self.prev_player_health > self.prev_opponent_health:
            # print('You win!')
            return 200
        else:
            # print('You lose!')
            return -200
        
    def _get_reward(self):
        player_health_area = self.game_screen_gray[15:20, 32:120]
        oppoent_health_area = self.game_screen_gray[15:20, 136:224]
        
        # Get health points using the number of pixels above 129.
        player_health = np.sum(player_health_area > 129) / player_health_area.size
        opponent_health = np.sum(oppoent_health_area > 129) / oppoent_health_area.size

        player_health_diff = self.prev_player_health - player_health
        opponent_health_diff = self.prev_opponent_health - opponent_health

        reward = (opponent_health_diff - player_health_diff) * 100 # max would be 100

        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health

        # Print the health values of the player and the opponent
        # print("Player health: %f Opponent health:%f" % (player_health, opponent_health))
        return reward

    def reset(self):
        observation = self.env.reset()
        self.prev_player_health = 1.0
        self.prev_opponent_health = 1.0
        return self._preprocess_observation(observation)

    def step(self, action):
        # observation, _, _, info = self.env.step(action)
        observation, _reward, _done, info = self.env.step(action)
        custom_reward = self._get_reward()

        custom_done = False
        if self.prev_player_health <= 0.00001 or self.prev_opponent_health <= 0.00001:
            custom_reward += self._get_win_or_lose_bonus()
            if not self.testing:
                custom_done = True
            else:
                self.prev_player_health = 1.0
                self.prev_opponent_health = 1.0
             
        return self._preprocess_observation(observation), custom_reward, custom_done, info