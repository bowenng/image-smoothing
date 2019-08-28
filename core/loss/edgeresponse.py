import torch
from torch import nn
from torch.functional import F


class EdgeResponse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images):
        edge_response = self.edge_response_kernel(images)
        return edge_response

    def edge_response_kernel(self, images):
        """
        1. reflection pads the images
        2. use the kernel below to calculate edge response

        kernel used to calculate edge response:
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        :return: Tensor of the same shape of image
        """

        # reflection pads the images
        reflection_pad = nn.ReflectionPad2d(2)
        images = reflection_pad(images)

        # use the kernel below to calculate edge response
        n_channels = images.shape[1]

        kernel_weights = torch.Tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

        # repeat the weights to match
        kernel_weights = kernel_weights.view(1, 1, 3, 3).repeat(1, n_channels, 1, 1)

        output = F.conv2d(images, kernel_weights)

        return output