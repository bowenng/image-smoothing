from torch.utils import data
import os
from PIL import Image


class Dataset(data.Dataset):

    def __init__(self, image_dir, edge_dir, edge_prefix="edge", transform=None):
        super().__init__()
        self.image_dir = image_dir
        image_files = os.listdir(image_dir)
        self.image_files = [image_file for image_file in image_files if self.is_image(image_file)]

        self.edge_dir = edge_dir
        self.edge_prefix = edge_prefix

        self.transform = transform

    @staticmethod
    def is_image(file):
        return file.lower().endswith(('.jpg', '.png', '.jpeg'))

    def __getitem__(self, index):
        image_file_name = self.image_files[index]
        edge_file_name = self.edge_prefix + image_file_name

        image_path = os.join(self.image_dir, image_file_name)
        edge_path = os.join(self.edge_dir, edge_file_name)

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path)

        if self.transform:
            image = self.transform(image)

        return (image, edge)

    def __len__(self):
        return len(self.image_files)

