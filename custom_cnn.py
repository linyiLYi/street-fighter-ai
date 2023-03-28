import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import mobilenet_v3_small

# Custom feature extractor (CNN)
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(CustomCNN, self).__init__(observation_space, features_dim=512)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16384, self.features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations.permute(0, 3, 1, 2))  # Swap the channel dimension
    