import os
from PIL import Image
import matplotlib.pyplot as plt
import  numpy as np

def display_image_folder(folder_name, start_idx=0, stop_idx=None):
    """
    Display all images in a given folder

    Args:
        folder_name : the folder name
        start_idx : index to start listing
        stop_idx : index to stop listing
    Returns:

    """
    list_image_names = os.listdir(folder_name)[start_idx:stop_idx]

    for img_name in list_image_names:
        img = Image.open(os.path.join(folder_name, img_name))
        img = np.array(img)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()