import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.data.data import EvalDataset
from core.network.smoothingnetwork import SmoothingNet

from torch.utils.data import DataLoader
import torch

import numpy as np
from PIL import Image

def play(output_images, output_dir, file_name):
    fig = plt.figure()


    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = [[plt.imshow(img, animated=True)] for img in output_images]

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(os.path.join(output_dir, file_name))
    plt.show()

def process_video(video_dir):

    ds = EvalDataset(video_dir)
    dataloader = DataLoader(ds, batch_size=1)

    net = SmoothingNet()
    state_dict = torch.load('model.pth', map_location='cpu')
    net.load_state_dict(state_dict['state_dict'])

    outputs = []

    for images in dataloader:
        out = net(images)
        outputs.extend(to_numpy(out))

    return outputs

def to_numpy(tensor):
    images = tensor.detach().cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = list(images)
    return images

if __name__ == '__main__':
    root = os.path.join('DAVIS', 'JPEGImages', '480p')
    folder = 'bear'
    video_dir = os.path.join(root, folder)
    output_dir = 'videos'
    outputs = process_video(video_dir)
    play(outputs, output_dir, folder+'.mp4')
