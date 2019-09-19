import torch
from torch import nn
from core.loss.dataloss import DataLoss
from core.loss.edgepreservingloss import EdgePreservingLoss
from core.loss.smoothnessloss import SmoothnessLoss


class TotalLoss(nn.Module):
    def __init__(self, w_s=1, w_e=0.1):
        super().__init__()
        self.w_s = w_s
        self.w_e = w_e
        self.data_loss = DataLoss()
        self.edge_preserving_loss = EdgePreservingLoss()
        self.smoothness_loss = SmoothnessLoss()
        self.E = 0.0
        self.D = 0.0
        self.S = 0.0

    def forward(self, original_images, smooth_images, binary_mask):
        self.E = self.w_e * self.edge_preserving_loss(binary_mask, original_images, smooth_images)
        self.D = self.data_loss(original_images, smooth_images)
        self.S = self.w_s * self.smoothness_loss(original_images, smooth_images)
        return self.D + self.E + self.S

    def get_loss(self):
        return {"D": self.D,
                "E": self.E,
                "S": self.S}