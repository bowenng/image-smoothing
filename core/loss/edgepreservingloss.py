import torch
from torch.functional import F
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class EdgePreservingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = EdgeResponse()
        self.L2_loss = nn.MSELoss()

    def forward(self, binary_mask, original_image, smooth_image):
        number_of_important_edges = binary_mask.sum(3).sum(2).sum(1)

        original_image_edge_response = self.E(original_image)
        smooth_image_edge_response = self.E(smooth_image)

        return torch.sum(1 / number_of_important_edges * binary_mask * \
               (original_image_edge_response - smooth_image_edge_response) ** 2)








