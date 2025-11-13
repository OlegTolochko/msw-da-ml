import math
import torch
import torch.nn as nn

class GaussianNLL(nn.Module):
    def __init__(self, logvar_min: float = -10.0, logvar_max: float = 5.0):
        super().__init__()
        self.log2pi = math.log(2 * math.pi)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, y, mu, log_var):

        log_var = torch.clamp(log_var, self.logvar_min, self.logvar_max)

        err = y - mu
        nll = 0.5 * (self.log2pi + log_var + torch.exp(-log_var) * err**2)
        loss = nll.mean()
        return loss