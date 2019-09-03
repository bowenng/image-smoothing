import torch
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class SmoothnessLoss(nn.Module):
    def __init__(self, sigma_color=0.1,sigma_space=7, lp=0.8, c1=20, c2=10, window_size=10, image_size=224):
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
        self.c1 = c1
        self.c2 = c2
        self.reflection_pad = nn.ReflectionPad2d(window_size)
        self.edge_response_calculator = EdgeResponse()
        self.unfold = nn.Unfold(kernel_size=image_size, padding=window_size)

    def forward(self, original_images, smooth_images):
        """
        :param original_image:
        :param smooth_image:
        :return:

        1. calculate Ti-Tj (batch_size, window_area, n_channel, height, width)
        2. calculate binary masks for (ws, p_large) and (wr, p_small), and calculate ws and wr
        3. apply the binary masks, and apply (w, p)
        """
        ti_minus_tj = self.calculate_ti_minus_tj(smooth_images)
        use_p_large_ws, use_p_small_wr = self.calculate_w_p_masks(original_images, smooth_images)
        ws = self.calculate_ws(smooth_images)
        wr = self.calculate_wr(original_images)

        shape = original_images.shape
        n_pixels = shape[2]*shape[3]
        batch_size = shape[0]
        smooth_loss = (1/(n_pixels*batch_size)) * torch.sum(use_p_large_ws * ws * (ti_minus_tj**2) + use_p_small_wr * wr * (ti_minus_tj**self.lp))
        return smooth_loss

    def calculate_ti_minus_tj(self, smooth_images):
        bs, c, h, w = smooth_images.shape
        image_patches = self.unfold(smooth_images).view(bs, -1, c, h, w)
        mask = image_patches > 0
        ti_minus_tj = mask * torch.abs(image_patches - smooth_images.view(bs, 1, c, h, w))
        return ti_minus_tj

    def calculate_w_p_masks(self, original_images, smooth_images):
        edge_response_original = self.edge_response_calculator(original_images)
        edge_response_smooth = self.edge_response_calculator(smooth_images)

        use_p_large_ws = (edge_response_original < self.c1) & \
                             ((edge_response_smooth - edge_response_original) > self.c2)

        use_p_small_wr = ~use_p_large_ws

        return use_p_large_ws.unsqueeze(1).float(), use_p_small_wr.unsqueeze(1).float()

    def calculate_wr(self, original_images):
        bs, c, h, w = original_images.shape
        image_patches = self.unfold(original_images)
        mask = image_patches > 0

        color_difference = image_patches - original_images.view(bs, 1, c, h, w)
        color_difference = mask * color_difference ** 2
        color_difference = color_difference.sum(axis=2)
        color_affinity = torch.exp(-1 * self.sigma_color * color_difference)
        return color_affinity

    def calculate_ws(self, smooth_images):
        window_size = self.window_size
        window_length = 2*window_size + 1

        batch_size, n_channel, height, width = smooth_images.shape
        ws = smooth_images.new(batch_size, window_length ** 2, height, width)

        for x in range(-window_size, window_size + 1):
            for y in range(-window_size, window_size + 1):
                x_y_1d_offset = x + window_size + (y + window_size) * window_length
                ws[:, x_y_1d_offset, :, :] = torch.exp(
                    torch.tensor([-1*self.sigma_space*(x**2 + y**2)]).view(1,1,1).repeat(batch_size, height, width)
                )
        return ws.view(batch_size, window_length * window_length, 1, height, width).repeat(1, 1, n_channel, 1, 1)


















