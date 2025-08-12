import torch
import torch.nn as nn
import torch.nn.functional as F

from core.settings import load_settings

settings = load_settings()
loss_config = settings.loss_config


class RMSEBiasLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, y_true, y_pred):
        # RMSE
        rmse_loss = torch.mean(
            torch.sqrt(torch.mean(torch.square(y_true - y_pred), axis=2) + self.eps)
        )

        # Bias Loss, punishes incorrect totals for a specified variable index
        y_true_variable_mean = torch.mean(y_true, dim=2)[
            :, loss_config.variable_to_punish_idx
        ]
        y_pred_variable_mean = torch.mean(y_pred, dim=2)[
            :, loss_config.variable_to_punish_idx
        ]
        bias_loss = torch.mean(
            torch.sqrt(torch.square(y_true_variable_mean - y_pred_variable_mean))
        )

        loss = rmse_loss + loss_config.bias_loss_weight * bias_loss

        return loss


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float):
    err = y_true - y_pred

    loss = torch.where(err > 0, tau * (err), (1 - tau) * (-err))
    return torch.mean(loss)
