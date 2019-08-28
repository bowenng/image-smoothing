import torch
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class SmoothnessLoss(nn.Module):
    def __init__(self, sigma_color,sigma_space,window_size=1, lp=0.8, c1=1, c2=1):
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
        window_length = 2*window_size+1
        # initialize Tensor of shape (batch size, window_size ** 2, n_channels, h, w) to hold Ti - Tj
        shape = smooth_images.shape
        batch_size, n_channel, height, width = shape[0], shape[1], shape[2], shape[3]
        ti_minus_tj = torch.Tensor(batch_size, window_length**2, n_channel, height, width)

        # reflection pads the image
        reflection_pad = nn.ReflectionPad2d(window_size)
        smooth_images_padded = reflection_pad(smooth_images)

        # loop through all surrounding pixel j's, and store the result in ti_minus_tj
        for x in range(-window_size, window_size+1):
            for y in range(-window_size, window_size+1):
                x_start = window_size + x
                x_end = x_start + width
                y_start = window_size + y
                y_end = y_start + height
                x_y_1d_offset = x + window_size + (y + window_size) * window_length
                ti_minus_tj[:,x_y_1d_offset, :, :, :] = torch.abs(smooth_images - smooth_images_padded[:, :,x_start:x_end, y_start:y_end])

        return ti_minus_tj

    def calculate_w_p_masks(self, original_images, smooth_images):
        edge_response_calculator = EdgeResponse()

        edge_response_original = edge_response_calculator(original_images)
        edge_response_smooth = edge_response_calculator(smooth_images)

        use_p_large_ws = (edge_response_original < self.c1) & \
                             ((edge_response_smooth - edge_response_original) > self.c2)

        use_p_small_wr = ~use_p_large_ws

        return use_p_large_ws, use_p_small_wr

    def calculate_wr(self, original_images):
        window_size = 1
        window_length = 2 * window_size + 1
        # initialize Tensor of shape (batch size, window_size ** 2, h, w) to hold Ti - Tj
        shape = original_images.shape
        batch_size, n_channel, height, width = shape[0], shape[1], shape[2], shape[3]
        wr = torch.Tensor(batch_size, window_length ** 2, height, width)
        # reflection pads the image
        reflection_pad = nn.ReflectionPad2d(window_size)
        original_images_padded = reflection_pad(original_images)

        for x in range(-window_size, window_size + 1):
            for y in range(-window_size, window_size + 1):
                x_start = window_size + x
                x_end = x_start + width
                y_start = window_size + y
                y_end = y_start + height
                x_y_1d_offset = x + window_size + (y + window_size) * window_length
                difference = original_images - original_images_padded[:, :, x_start:x_end, y_start:y_end]
                wr[:, x_y_1d_offset, :, :] = torch.exp((-self.sigma_color * difference**2).sum(1))
        return wr.view(batch_size, window_length * window_length, 1, height, width).repeat(1, 1, n_channel, 1, 1)


















