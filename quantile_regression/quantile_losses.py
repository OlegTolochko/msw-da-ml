import torch

def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float):
    err = y_true - y_pred

    loss = torch.where(err > 0, tau * (err), (1 - tau) * (-err))
    return torch.mean(loss)
