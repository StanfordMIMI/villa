import torch
import abc


class BaseLoss(torch.nn.Module):
    __metaclass__ = abc.ABC

    def __init__(self):
        super().__init__()

        self.iteration = 0
        self.running_loss = 0
        self.mean_running_loss = 0

    def forward(self, input):
        return input

    def update_running_loss(self, loss):
        self.iteration += 1
        self.running_loss += loss.item()
        self.mean_running_loss = self.running_loss / self.iteration
