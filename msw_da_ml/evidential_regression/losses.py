import math
import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    def __init__(
        self, beta: float = 0.5, logvar_min: float = -6.0, logvar_max: float = 2.0
    ):
        super().__init__()
        self.beta = beta
        self.log2pi = math.log(2 * math.pi)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, y, mu, log_var):
        log_var = torch.clamp(log_var, self.logvar_min, self.logvar_max)

        err = y - mu
        precision = torch.exp(-log_var)

        precision_detached = precision.detach()

        precision_weighted = (
            precision_detached ** (1 - self.beta) * precision**self.beta
        )

        nll = 0.5 * (self.log2pi + log_var + precision_weighted * err**2)

        return nll.mean()


class NIGLoss(nn.Module):
    def __init__(self, reg_coef: float = 1.0):
        super().__init__()
        self.reg_coef = reg_coef

    def forward(self, gamma, nu, alpha, beta, y):
        omega = 2 * beta * (1 + nu)
        L_nll = (
            1 / 2 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 1 / 2) * torch.log((y - gamma) ** 2 * nu + omega)
            + torch.log(
                torch.lgamma(alpha)
                / torch.lgamma(alpha + 1 / 2)
            )
        )
        L_r = torch.abs(y - gamma) * (2 * nu + alpha)
        loss = torch.mean(L_nll + self.reg_coef * L_r)
        return loss
