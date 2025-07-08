import torch.functional as F
import torch.nn as nn

from settings import load_settings

settings = load_settings()
network_config = settings.network_config


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        layers += [
            nn.Conv1d(
                in_channels=network_config.in_channels,
                out_channels=network_config.hidden_channels,
                kernel_size=network_config.kernel_size,
                padding=network_config.kernel_size // 2,
                padding_mode="circular",
            ),
            nn.SELU(),
        ]

        for i in range(network_config.num_layers - 2):
            layers += [
                nn.Conv1d(
                    in_channels=network_config.hidden_channels,
                    out_channels=network_config.hidden_channels,
                    kernel_size=network_config.kernel_size,
                    padding=network_config.kernel_size // 2,
                    padding_mode="circular",
                ),
                nn.SELU(),
            ]

        layers += [
            nn.Conv1d(
                in_channels=network_config.hidden_channels,
                out_channels=3,
                kernel_size=network_config.kernel_size,
                padding=network_config.kernel_size // 2,
                padding_mode="circular",
            ),
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x[:, 2] = F.ReLU(x[:, 2])

        return x
