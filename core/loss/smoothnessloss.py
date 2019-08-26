import torch
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class SmoothnessLoss(nn.Module):
    def __init__(self, c1=1.0, c2=2.0, p_small=0.8, p_large=2.0):
        super().__init__()
        self.E = EdgeResponse()
        self.c1 = c1
        self.c2 = c2
        self.p_small = p_small
        self.p_large = p_large

    def forward(self, original_image, smooth_image):
        smooth_image_without_border = smooth_image[:, :, 1:-1, 1:-1]
        original_image_without_border = original_image[:, :, 1:-1, 1:-1]
        width = smooth_image.shape[2]
        height = smooth_image.shape[3]

        Ti_Tj = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                Ti_Tj.append(
                    torch.abs(
                        torch.sum(
                            smooth_image[:, :, 1+x:width-1+x, 1+y:height-1+y]-smooth_image_without_border, dim=1)))

        Ti_Tj_tensor = torch.stack(Ti_Tj, dim=1)

        original_image_edge_response = self.E(original_image)
        smooth_image_edge_response = self.E(smooth_image)

        p_large_mask = (original_image_edge_response < self.c1) \
                          & (smooth_image_edge_response - original_image_edge_response > self.c2)
        p_small_mask = ~p_large_mask

        #wr
        adjacent_differences = []
        # loop through every pixel j around the pixel i, and calculate the absolute difference
        for x in range(-1, 2):
            for y in range(-1, 2):
                adjacent_differences.append(
                    torch.sum((original_image[:, :, 1 + x:width - 1 + x, 1 + y:height - 1 + y] -
                               original_image_without_border) ** 2, dim=1))
        adjacent_differences = torch.stack(adjacent_differences, dim=1)
        wr = torch.exp(-adjacent_differences)

        num_pixels = width*height

        return torch.sum(1 / num_pixels * (
            wr * p_large_mask.float() * Ti_Tj_tensor ** self.p_small
            + wr * p_small_mask.float() * Ti_Tj_tensor ** self.p_large
        ))














