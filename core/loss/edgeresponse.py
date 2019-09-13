import torch
from torch import nn
from torch.functional import F


class EdgeResponse(nn.Module):
    def __init__(self, image_size=224, window_size=1):
        super().__init__()
        self.unfold = nn.Unfold(image_size, padding=1)

    def forward(self, images):
        # 1. reflection pad the images
        # 2. calculate simple edge response in a 3x3 window
        bs, c, h, w = images.shape

        image_patches = self.unfold(images).transpose(1, 2).view(bs, -1, c, h, w).detach()
        mask = image_patches > 0.0
        i_minus_j = torch.where(mask, image_patches - images.view(bs, 1, c, h, w),torch.zeros_like(image_patches))
        i_minus_j = torch.abs(i_minus_j)
        i_minus_j = i_minus_j.sum(2)
        edge_response = i_minus_j.sum(1)
        return edge_response


