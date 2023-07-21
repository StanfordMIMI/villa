import torch
import numpy as np
from villa.losses.default import BaseLoss


class Stage1_Loss(BaseLoss):
    def __init__(self, temp: float, one_proj: bool, data_dir: str):
        super(Stage1_Loss, self).__init__()

        self.temperature = temp
        self.one_proj = one_proj

        attr_to_embs = torch.load(f"{data_dir}/attr_embs.pth")
        self.attr_embs = []
        for a in attr_to_embs:
            self.attr_embs.append(attr_to_embs[a])
        self.attr_embs = torch.tensor(np.stack(self.attr_embs)).cuda().to(torch.float32)

    def forward(self, pred, sample):
        pass
