from torch import nn


class DataLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L2_norm = nn.MSELoss()

    def forward(self, original_image, smooth_image):
        return self.L2_norm(original_image, smooth_image)
