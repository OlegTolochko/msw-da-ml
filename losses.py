import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSEBiasLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, y_true, y_pred):
        loss = (
            torch.mean(
                torch.sqrt(torch.mean(torch.square(y_true - y_pred), axis=1)), axis=-1
            )
            + self.eps
        )

        return loss
