import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class FAMOLightning(nn.Module):
    """
    Fast Adaptive Multitask Optimization (Lightning/PyTorch対応版)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        super().__init__()
        self.min_losses = torch.zeros(n_tasks, device=device)
        self.w = nn.Parameter(torch.zeros(n_tasks, device=device))  # nn.Parameterで管理
        self.gamma = gamma
        self.w_lr = w_lr
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w.grad = d

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            list, torch.Tensor
        ] = None,
    ) -> Union[torch.Tensor, None]:
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss 