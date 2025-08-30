import torch.nn.functional as F
import torch.nn as nn

from msw_da_ml.settings import load_settings

settings = load_settings()
network_config = settings.network_config


class QuantileCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
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

        self.lower_projection = nn.Conv1d(
            in_channels=network_config.hidden_channels,
            out_channels=3,
            kernel_size=network_config.kernel_size,
            padding=network_config.kernel_size // 2,
            padding_mode="circular",
        )

        self.higher_projection = nn.Conv1d(
            in_channels=network_config.hidden_channels,
            out_channels=3,
            kernel_size=network_config.kernel_size,
            padding=network_config.kernel_size // 2,
            padding_mode="circular",
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x_lower = self.lower_projection(x)
        x_higher = self.higher_projection(x)

        return x_lower, x_higher
