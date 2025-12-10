import os
import cv2
from PIL import Image
import numpy as np
from pillow_heif import register_heif_opener

main_directory = 'data/occlusion'
original_directory = os.path.join(main_directory, 'original')
resized_directory = os.path.join(main_directory, 'resized_original')

def create_resized_directories():
    if not os.path.exists(resized_directory):
        os.makedirs(resized_directory)
    register_heif_opener()

    for filename in os.listdir(original_directory):
        img = Image.open(os.path.join(original_directory, filename))
        img = np.array(img.convert('RGB'))
        img = cv2.resize(img, (299, 299))
        cv2.imwrite(os.path.join(resized_directory, filename.split('.')[0] + '.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    create_resized_directories()