import torch
from torch import nn


class EdgeResponse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        # cut the border of the image
        image_without_border = image[:, :, 1:-1, 1:-1]

        # get metadata needed to calculate edge response
        image_shape = image.shape
        width = image_shape[2]
        height = image_shape[3]

        # calculate the edge response in a 3x3 window
        adjacent_differences = []
        # loop through every pixel j around the pixel i, and calculate the absolute difference
        for x in range(-1, 2):
            for y in range(-1, 2):
                adjacent_differences.append(torch.abs(torch.sum(image[:, :, 1+x:width-1+x, 1+y:height-1+y]-image_without_border, dim=1)))

        # sum over pixel j
        edge_response = torch.stack(adjacent_differences, dim=0).sum(dim=0).unsqueeze(1)
        return edge_response
