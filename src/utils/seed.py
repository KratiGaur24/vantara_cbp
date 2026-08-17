import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
  """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
  """Detect and return target backend hardware device (CPU or CUDA)."""
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")