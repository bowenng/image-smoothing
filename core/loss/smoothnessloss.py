import torch
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class SmoothnessLoss(nn.Module):
    def __init__(self, sigma_color,sigma_space,window_size=1, lp=0.8):
        """

        :param sigma_color: see paper
        :param sigma_space: see paper
        :param window_size: distance from center, =1 for 3x3 kernel, =2 for 5x5 kernel
        :param lp: LP norm
        """
        super().__init__()
        self.sigma_color = 1 / (sigma_color * sigma_color * 2)
        self.sigma_space = 1 / (sigma_space * sigma_space * 2)
        self.window_size = window_size
        self.lp = lp

    def forward(self, original_images, smooth_images):
        """
        :param original_image:
        :param smooth_image:
        :return:

        1. calculate Ti-Tj and stack the resulting Tensors
        2. calculate binary masks for (ws, p_large) and (wr, p_small), and calculate ws and wr
        3. apply the binary masks, and apply (w, p)
        """

    def calculate_ti_minus_tj(self, smooth_images):
        window_size = self.window_size

        # initialize Tensor of shape (batch size, window_size ** 2, h, w) to hold Ti - Tj
        shape = smooth_images.shape
        batch_size, height, width = shape[0], shape[2], shape[3]
        ti_minus_tj = torch.Tensor(batch_size, window_size*window_size, height, width)

        # reflection pads the image
        reflection_pad = nn.ReflectionPad2d(1)
        smooth_images_padded = reflection_pad(smooth_images)

        # loop through all surrounding pixel j's, and store the result in ti_minus_tj
        for x in range(-window_size, window_size+1):
            for y in range(-window_size, window_size+1):
                ti_minus_tj[:,x+y, :, :] = smooth_images - smooth_images_padded[:, :,window_size+x, window_size+y]

        return ti_minus_tj




















