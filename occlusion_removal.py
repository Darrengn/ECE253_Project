import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
'''
BEFORE RUNNING THIS SCRIPT:
Run create_resized.py to get the 299x299 versions of the original images from data/occulsions/original
    - Creates the data/occulsions/resized_original directory

This script will create two directories from the resized_original folder.
    1) pngs that are 299x299 resized verisons of the ogs after bilateral filtering
    2) pngs that are 299x299 resized versions of the ogs after DnCNN

The comparison for orignal vs bilateral vs dncnn will be done in occlusion_removal_comparison.py
'''

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        layers = []
        #First Layer = Conv + ReLU
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        #Next 18 Layers
        for _ in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))

        #Residual Noise Layer
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


def dncnn_removal(model, img_path):
    img = Image.open(img_path).convert('RGB')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
    transforms.Resize((299, 299)),             
    #transforms.CenterCrop(299),         
    transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
            noise_prediction = model(img_tensor)
            denoised_img = img_tensor - noise_prediction
            denoised_img = torch.clamp(denoised_img, 0, 1)
    denoised_img = denoised_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return denoised_img



occulsion_data_directory = 'data/occlusion/'

resized_directory = os.path.join(occulsion_data_directory, 'resized_original')
bilateral_directory = os.path.join(occulsion_data_directory, 'bilateral_filtered')
dncnn_directory = os.path.join(occulsion_data_directory, 'dncnn')
median_directory = os.path.join(occulsion_data_directory, 'median')

if __name__ == '__main__':
    if not os.path.exists(bilateral_directory):
        os.makedirs(bilateral_directory)
    if not os.path.exists(dncnn_directory):
        os.makedirs(dncnn_directory)
    if not os.path.exists(median_directory):
        os.makedirs(median_directory)
    do_dncnn = False
    do_median = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dncnn = DnCNN()
    dncnn.to(device)
    state_dict = torch.load('dncnn_py.pth')
    dncnn.load_state_dict(state_dict)
    dncnn.eval()

    for filename in os.listdir(resized_directory):
        img_path = os.path.join(resized_directory, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bilateral_img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)    #11 with 19, 75, 50-60
        bilateral_path = os.path.join(bilateral_directory, filename.split('.')[0] + '_bilateral.png')
        cv2.imwrite(bilateral_path, cv2.cvtColor(bilateral_img, cv2.COLOR_RGB2BGR))


        if do_median:
            median_img = cv2.medianBlur(img, 5) #12 with 5, 5 with 7, 7 with 3
            median_path = os.path.join(median_directory, filename.split('.')[0] + '_median.png')
            cv2.imwrite(median_path, cv2.cvtColor(median_img, cv2.COLOR_RGB2BGR))
        if do_dncnn:
            dncnn_img = dncnn_removal(dncnn, img_path)
            dncnn_path = os.path.join(dncnn_directory, filename.split('.')[0] + '_dncnn.png')
            cv2.imwrite(dncnn_path, cv2.cvtColor((dncnn_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


