import numpy as np


def calculate_empirical_quantile(scores: np.ndarray, alpha: float):
    """
    Calculate empirical quantile from scores.

    Returns:
        Quantile values of shape (time,)
    """
    if scores.ndim == 3:
        # Shape: (seeds, time, grid)
        num_seeds, num_time, num_grid = scores.shape
        # Reshape to (seeds * grid, time)
        s = scores.transpose(0, 2, 1).reshape(num_seeds * num_grid, num_time)
    elif scores.ndim == 4:
        # Shape: (seeds, time, grid, ens)
        num_seeds, num_time, num_grid, num_ens = scores.shape
        # Reshape to (seeds * grid * ens, time)
        s = scores.transpose(0, 2, 3, 1).reshape(
            num_seeds * num_grid * num_ens, num_time
        )
    else:
        raise ValueError(f"Unexpected scores shape: {scores.shape}")

    m = s.shape[0]
    k = int(np.ceil((1.0 - alpha) * (m + 1))) - 1
    k = np.clip(k, 0, m - 1)
    s_part = np.partition(s, k, axis=0)
    return s_part[k]  # Shape: (time,)
