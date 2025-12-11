import torch.nn.functional as F
import torch.nn as nn

from msw_da_ml.settings import load_settings

settings = load_settings()
network_config = settings.network_config


class NIGCNNModel(nn.Module):
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

        layers += [
            nn.Conv1d(
                in_channels=network_config.hidden_channels,
                out_channels=12,
                kernel_size=network_config.kernel_size,
                padding=network_config.kernel_size // 2,
                padding_mode="circular",
            ),
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        gamma, nu, alpha, beta = x[:, :3], x[:, 3:6], x[:, 6:9], x[:, 9:]
        nu = F.softplus(nu)
        beta = F.softplus(beta)
        alpha = F.softplus(alpha) + 1

        return gamma, nu, alpha, beta
