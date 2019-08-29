from torchvision import transforms


def image_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(),
        transforms.Normalize(),
        transforms.ToTensor()
    ])


def edge_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(),
        transforms.Normalize(),
        transforms.ToTensor()
    ])


def is_image(file):
    return file.lower().endswith(('.jpg', '.png', '.jpeg'))