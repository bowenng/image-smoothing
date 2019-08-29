import torch
from torch import nn
from core.loss.dataloss import DataLoss
from core.loss.edgepreservingloss import EdgePreservingLoss
from core.loss.smoothnessloss import SmoothnessLoss


class TotalLoss(nn.Module):
    def __init__(self, w_smooth=1, w_edge_preserving=0.1):
        super().__init__()
        self.w_smooth = w_smooth
        self.w_edge_preserving = w_edge_preserving
        self.data_loss = DataLoss()
        self.edge_preserving_loss = EdgePreservingLoss()
        self.smoothness_loss = SmoothnessLoss()

    def forward(self, original_images, smooth_images):
        return self.data_loss(original_images, smooth_images) + self.w_smooth * self.smoothness_loss(original_images, smooth_images) + self.w_edge_preserving + self.edge_preserving_loss(original_images, smooth_images)
