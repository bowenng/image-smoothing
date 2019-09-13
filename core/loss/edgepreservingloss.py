import torch
from torch.functional import F
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class EdgePreservingLoss(nn.Module):
    def __init__(self, image_size=224, window_size=10):
        super().__init__()
        self.E = EdgeResponse(image_size, window_size)
        self.L2_loss = nn.MSELoss()

    def forward(self, binary_mask, original_image, smooth_image):
        binary_mask = binary_mask > 0

        original_image_edge_response = self.E(original_image).detach()
        smooth_image_edge_response = self.E(smooth_image)

        normalization_factor = 1 / torch.sum(binary_mask.float())

        difference = (original_image_edge_response - smooth_image_edge_response) ** 2
        return normalization_factor * torch.sum(torch.where(binary_mask, difference, torch.zeros_like(difference)))

