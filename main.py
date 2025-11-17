import numpy as np
import cv2
import matplotlib.pyplot as plt
from bright_adjust import *

img = cv2.imread('test_img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299))
output = anhe(img)

plt.imshow(img)
plt.show()