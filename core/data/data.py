from torch.utils import data
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
    def __init__(self, image_dir, image_transform=eval_transform()) -> None:
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
    def __init__(self, video_dir, n_neighbors) -> None:
        super().__init__()
        assert n_neighbors % 2 == 1, "n_neighbors must be odd."
        # video files shape: n_files, n_frames
        video_files = [sorted(list(filter(is_image,os.listdir(os.path.join(video_dir,video_file)))))
                            for video_file in os.listdir(video_dir)]

        min_frame = float('inf')
        for file in video_files:
            min_frame = min(min_frame, len(file))

        self.video_files = [file[:min_frame] for file in video_files]
        self.n_videos = len(self.video_files)
        self.n_frames = min_frame
        self.n_neighbors = n_neighbors

    def __getitem__(self, index):
        video_idx = index % self.n_frames
        return self.video_files[video_idx][index:index+2*self.n_neighbors]

    def __len__(self):
        return self.n_videos * (self.n_frames-2*self.n_neighbors)

