from torch.utils import data
import os
from PIL import Image
from core.data.utils import image_transform, edge_transform, is_image


class Dataset(data.Dataset):

    def __init__(self, image_dir, edge_dir, edge_prefix="edge_", image_transform=image_transform, edge_transform=edge_transform):
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

        return image, edge[:,1,:,:]

    def __len__(self):
        return len(self.image_files)

