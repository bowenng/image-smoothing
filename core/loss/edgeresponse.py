import torch
from torch import nn
from torch.functional import F


class EdgeResponse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images):
        # 1. reflection pad the images
        # 2. calculate simple edge response in a 3x3 window
        shape = images.shape
        batch_size, n_channel, height, width = shape

        # reflection pads the images
        reflection_pad = nn.ReflectionPad2d(2)
        images_padded = reflection_pad(images)

        # 3x3 window
        window_size = 1
        window_length = 3

        i_minus_j = torch.Tensor(batch_size, window_length**2, height, width)

        for x in range(-window_size, window_size + 1):
            for y in range(-window_size, window_size + 1):
                x_start = window_size + x
                x_end = x_start + width
                y_start = window_size + y
                y_end = y_start + height
                x_y_1d_offset = x + window_size + (y + window_size) * window_length
                i_minus_j[:, x_y_1d_offset, :, :] = torch.abs((images - images_padded[:, :, x_start:x_end, y_start:y_end]).sum(1))

        edge_response = i_minus_j.sum(1).view(batch_size, 1, height, width)
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