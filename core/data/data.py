from torch.utils import data
import torch
from torchvision import transforms
import os
from PIL import Image
from core.data.utils import image_transform, edge_transform, eval_transform, is_image


class Dataset(data.Dataset):
    def __init__(self, image_dir, edge_dir, edge_prefix="edge_", image_transform=image_transform(), edge_transform=edge_transform()):
        super().__init__()
        self.image_dir = image_dir
        image_files = os.listdir(image_dir)
        self.image_files = [image_file for image_file in image_files if is_image(image_file)]

        self.edge_dir = edge_dir
        self.edge_prefix = edge_prefix

        self.image_transform = image_transform
        self.edge_transform = edge_transform

    def __getitem__(self, index):
        image_file_name = self.image_files[index]
        edge_file_name = self.edge_prefix + image_file_name

        image_path = os.path.join(self.image_dir, image_file_name)
        edge_path = os.path.join(self.edge_dir, edge_file_name)

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.edge_transform:
            edge = self.edge_transform(edge)

        return image, edge

    def __len__(self):
        return len(self.image_files)


class EvalDataset(data.Dataset):
    def __init__(self, image_dir, image_transform=eval_transform()):
        super().__init__()
        self.image_dir = image_dir
        image_files = os.listdir(image_dir)
        self.image_files = sorted([image_file for image_file in image_files if is_image(image_file)])
        self.image_transform = image_transform

    def __getitem__(self, index):
        image_file_name = self.image_files[index]

        image_path = os.path.join(self.image_dir, image_file_name)

        image = Image.open(image_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        return image

    def __len__(self):
        return len(self.image_files)


class MultiFramesDataset(data.Dataset):
    def __init__(self, root_dir, n_neighbors) -> None:
        super().__init__()
        assert n_neighbors >= 0, "n_neighbors must be >= 0."
        # video files shape: n_files, n_frames
        self.root_dir = root_dir
        self.video_dir = "images"
        self.edge_dir = "edges"
        video_dir = os.path.join(root_dir, self.video_dir)
        edge_dir = os.path.join(root_dir, self.edge_dir)

        video_folders = [video_folder for video_folder in os.listdir(video_dir)]
        video_files = [sorted(list(filter(is_image,os.listdir(os.path.join(video_dir,video_folder)))))
                       for video_folder in video_folders]
        
        edge_folders = [edge_folder for edge_folder in os.listdir(edge_dir)]
        edge_files = [sorted(list(filter(is_image,os.listdir(os.path.join(edge_dir,edge_folder)))))
                      for edge_folder in edge_folders]

        min_frame = float('inf')
        for file in video_files:
            min_frame = min(min_frame, len(file))
        min_frame = min_frame - min_frame % (2*n_neighbors+1)
        self.video_folders = video_folders
        self.edge_folders = edge_folders
        
        self.video_files = [file[:min_frame] for file in video_files]
        self.edge_files = [file[:min_frame] for file in edge_files]
    
        self.n_videos = len(self.video_files)
        self.n_frames = min_frame
        self.n_neighbors = n_neighbors

        self.video_transform = transforms.Compose([
            transforms.Lambda(lambda frames: [transforms.CenterCrop(224)(frame) for frame in frames]),
            transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames]))
        ])
        self.edge_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        video_idx = index // self.n_videos
        frame_start = index // self.n_frames 
        frame_end = frame_start + self.n_neighbors * 2 + 1
        
        video_folder = self.video_folders[video_idx]
        edge_folder = self.edge_folders[video_idx]
        videos_files = self.video_files[video_idx][frame_start:frame_end]
        edge_file = self.edge_files[video_idx][frame_start:frame_end][self.n_neighbors]

        videos = [Image.open(os.path.join(self.root_dir, self.video_dir, video_folder, file)).convert("RGB") for file in videos_files]
        edge = Image.open(os.path.join(self.root_dir, self.edge_dir, edge_folder ,edge_file))

        videos = self.video_transform(videos)
        edge = self.edge_transform(edge)
        center_frame = videos[self.n_neighbors]

        return videos, edge, center_frame

    def __len__(self):
        return self.n_videos * (self.n_frames-2*self.n_neighbors-1)

