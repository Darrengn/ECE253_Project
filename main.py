import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
from bright_adjust import *


register_heif_opener()
img = Image.open("data/contrast/con1.HEIC")
img = np.array(img.convert('RGB'))
img = cv2.resize(img, (299, 299))

out = anhe(img)
out2 = contrast_adjust(img)

fig, axes = plt.subplots(2, 2)
axes[0,0].imshow(img)
axes[0,1].imshow(out)
axes[1,0].imshow(out2)
plt.show()

