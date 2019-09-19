import torch
from torch import nn
from core.loss.edgeresponse import EdgeResponse


class SmoothnessLoss(nn.Module):
    def __init__(self, sigma_color=0.1, sigma_space=7.0, lp=0.8, c1=10.0, c2=5.0, window_size=10, image_size=224,
                 alpha=5.0):
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
        self.alpha = alpha
        self.reflection_pad = nn.ReflectionPad2d(window_size)
        self.edge_response_calculator = EdgeResponse(image_size, window_size)
        self.unfold = nn.Unfold(kernel_size=image_size, padding=window_size)
        self.epsilon = 1e-6

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
        n_pixels = shape[2] * shape[3]
        batch_size = shape[0]

        scale_factor = (1 / (n_pixels * batch_size)) / (self.window_size ** 2)
        ws = torch.where(use_p_large_ws, ws, torch.zeros_like(ws))
        wr = torch.where(use_p_small_wr, wr, torch.zeros_like(wr))
        large_term = self.alpha * ws * ti_minus_tj ** 2
        small_term = wr * (ti_minus_tj + self.epsilon) ** self.lp
        smooth_loss = scale_factor * torch.sum(large_term + small_term)
        return smooth_loss

    def calculate_ti_minus_tj(self, smooth_images):
        bs, c, h, w = smooth_images.shape
        image_patches = self.unfold(smooth_images).transpose(1, 2).view(bs, -1, c, h, w).detach()
        mask = image_patches > 0
        ti_minus_tj = torch.abs(
            torch.where(mask, (image_patches - smooth_images.view(bs, 1, c, h, w)), torch.zeros_like(image_patches)))
        return ti_minus_tj

    def calculate_w_p_masks(self, original_images, smooth_images):
        edge_response_original = self.edge_response_calculator(original_images)
        edge_response_smooth = self.edge_response_calculator(smooth_images)

        use_p_large_ws = (edge_response_original < self.c1) & \
                         ((edge_response_smooth - edge_response_original) > self.c2)

        use_p_small_wr = ~use_p_large_ws
        return use_p_large_ws.unsqueeze(1).unsqueeze(1), use_p_small_wr.unsqueeze(1).unsqueeze(1)

    def calculate_wr(self, original_images):
        bs, c, h, w = original_images.shape
        image_patches = self.unfold(original_images).transpose(1, 2).view(bs, -1, c, h, w).detach()
        mask = image_patches > 0.0
        color_difference = image_patches - original_images.view(bs, 1, c, h, w)
        color_difference = color_difference ** 2
        color_difference = color_difference.sum(2, keepdim=True)
        color_affinity = torch.exp(-1.0 * self.sigma_color * color_difference)
        color_difference = torch.where(mask, color_difference, torch.zeros_like(color_difference))
        return color_affinity

    def calculate_ws(self, smooth_images):
        window_size = self.window_size
        h = torch.arange(-window_size, window_size + 1).view(-1, 1).repeat((1, 2 * window_size + 1))
        w = torch.arange(-window_size, window_size + 1).view(1, -1).repeat(2 * window_size + 1, 1)
        ws = torch.exp(-1.0 * self.sigma_space * (h ** 2 + w ** 2).float())
        ws = ws.view(1, -1, 1, 1, 1)
        return ws.cuda()
