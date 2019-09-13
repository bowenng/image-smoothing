import cv2 as cv
from core.data.utils import is_image
import os


def make_edge_files(image_dir, edge_dir, cwd, high=255, low=50):
    os.chdir(cwd)
    os.mkdir(edge_dir)
    image_files = os.listdir(image_dir)
    images = filter(is_image, image_files)

    for image_name in images:
        image = cv.imread(os.path.join(image_dir, image_name), 0)
        edges = cv.Canny(image, high, low)
        cv.imwrite(os.path.join(edge_dir, "edge_"+image_name), edges)



