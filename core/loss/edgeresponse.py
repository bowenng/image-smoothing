import torch
from torch import nn
from torch.functional import F


class EdgeResponse(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.unfold = nn.Unfold(image_size, padding=1)

    def forward(self, images):
        # 1. reflection pad the images
        # 2. calculate simple edge response in a 3x3 window
        bs, c, h, w = images.shape

        image_patches = self.unfold(images)

        i_minus_j = image_patches - images.view(bs, 1, c, h, w)

        edge_response = torch.abs(i_minus_j).sum(axis=2)
        return edge_response


