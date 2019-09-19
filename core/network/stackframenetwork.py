from core.network.smoothingnetwork import SmoothingNet
from torch import nn
import torch


class StackFrameNetwork(SmoothingNet):
    def __init__(self, n_neighbors=3):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_frames = 2 * n_neighbors + 1
        self.center = self.n_frames//2
        self.conv1_in = nn.Conv2d(in_channels=3*self.n_frames, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, image_stack):
        # image_stack: bs, n_frames * 3, h, w
        # center image = the image to output
        center_image = image_stack[:, self.center*3:self.center*3+3, :, :]
        smooth_image_residual = self.relu(self.bn1(self.conv1_in(image_stack)))
        smooth_image_residual = self.relu(self.bn2(self.conv2_in(smooth_image_residual)))
        smooth_image_residual = self.relu(self.bn3(self.conv3_downsample(smooth_image_residual)))

        smooth_image_residual = self.resnets(smooth_image_residual)
        smooth_image_residual = self.relu(self.bn4(self.transpose_conv1(smooth_image_residual)))
        smooth_image_residual = self.relu(self.bn5(self.conv1_out(smooth_image_residual)))
        smooth_image_residual = self.conv2_out(smooth_image_residual) / 255.0

        return center_image + smooth_image_residual
