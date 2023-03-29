import gym
import torch
import torchvision
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom MobileNetV3 Feature Extractor
class MobileNetV3Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(MobileNetV3Extractor, self).__init__(observation_space, features_dim=576)
        self.mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.mobilenet = torch.nn.Sequential(*list(self.mobilenet.children())[:-1])
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.mobilenet(observations)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x
