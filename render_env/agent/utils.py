import numpy as np
import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def to_gpu_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).cuda()


def gpu_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_cpu_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def cpu_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy()


def soft_update(network: torch.nn.Module, target_network: torch.nn.Module, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))
