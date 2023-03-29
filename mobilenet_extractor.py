import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import mobilenet_v3_small

# Custom MobileNetV3 Feature Extractor
class MobileNetV3Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(MobileNetV3Extractor, self).__init__(observation_space, features_dim=256)
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(576, self.features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # x = observations.permute(0, 2, 3, 1)  # Swap the channel dimension
        x = self.mobilenet.features(observations)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
