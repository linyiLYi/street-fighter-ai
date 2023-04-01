import torch.nn as nn

def conv2d_custom_init(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    nn.init.xavier_uniform_(conv.weight)
    return conv

def custom_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        conv2d_custom_init(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.Relu(),
        nn.MaxPool2d((2, 2))
    )

# Custom feature extractor (CNN)
class CustomCNN(nn.Module):
    def __init__(self, num_frames, num_moves, num_attacks):
        super(CustomCNN, self).__init__()
        self.num_moves = num_moves
        self.num_attacks = num_attacks
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=0),
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
        return self.cnn(observations)
    